import pandas as pd
import numpy as np
from sklearn import feature_selection
from collections import defaultdict
import os
import math
from sklearn.metrics import classification_report
from joblib import Parallel, delayed
from sklearn.base import TransformerMixin, BaseEstimator
from ..base_utils import RulerMixin

class Ripperk(BaseEstimator, TransformerMixin):
    """
    整体学习流程如下：
    1. 分割growset和prunset
    2. 对于growset， 运用irep学习规则集。
       a. 运用foil在growset上学习单规则（一系列条件交集）,
       b. 在prunset上进行剪枝
       c. 该规则加入到ruleset中，剔除该规则覆盖样本，余下样本继续学习，直到ruleset不满足mdl条件
    3. 在完整dataset上对ruleset进行优化
       最终学习到的ruleset是个无序规则集。试想下，先学到A rule，再学到B。那么即使互换A，B位置，B覆盖的样本要么是A中的一部分，要么就是余下样本中满足B条件的，换言之，打乱顺序最后覆盖到的样本是一样的。
    TODO: 如果依次学到A，B，C，只取两条，是不是选择A，B会更好呢？感觉并不是。甚至于不是其中任意两条的组合之一。一般会有这种需求，应该是因为学到规则覆盖样本的正例率不达到要求。那么应该通过设定条件的方式重新学习。
          比如在foil过程中，强制要求正例率达到一定阈值，不然直接给出负gain。good！确实应该加入该条件来进行控制。同理，对规则覆盖的样本量也可以最小限制。有时并不苛求样本量，那么foil gain计算规则的方法是否合适呢？
    """

    def __init__(self, prun_ratio=0.2, dl_threshold=64, k=2, sample_threshold=100, diff_threshold=0.0, q=10, n_jobs=-1):
        self.prun_ratio = prun_ratio
        self.dl_threshold = dl_threshold
        self.k = k
        self.sample_threshold = sample_threshold
        self.diff_threshold = diff_threshold
        self.q = q
        self.n_jobs = n_jobs

        self.grow_rule_iter_num = 0


    def fit(self, df, label):
        self.rulesets = {}

        self._get_conditions(df)

        items = list(label.value_counts().sort_values(ascending=False).index)
        self.items = list(items)

        while len(items) > 1:
            # get cls from end to start, from small to big
            item = items.pop()
            pos = df[label == item]
            neg = df[label != item]

            ruleset = self.irep(pos, neg)

            for _ in range(self.k):
                ruleset = self.optimize(pos, neg, ruleset)

            df = self.remove_cases(df, ruleset)

            self.rulesets[item] = ruleset

    def predict(self, df):
        labels = np.array([self.items[0]] * df.shape[0])

        index_bool = np.array([True] * df.shape[0])
        for item in self.items[1:][::-1]:
            item_bool = self.bindings(df, self.rulesets[item])
            item_bool &= index_bool
            labels[item_bool] = item
            index_bool &= ~item_bool

        return labels

    def irep(self, pos, neg):
        rule_set = []
        rule = {}

        min_dl = self.init_dl

        while pos.shape[0] > 0:
            print('rule:', rule, len(pos), len(neg))

            pos_chunk = int((1 - self.prun_ratio) * pos.shape[0])
            neg_chunk = int((1 - self.prun_ratio) * neg.shape[0])

            pos_grow = pos.iloc[:pos_chunk, :]
            neg_grow = neg.iloc[:neg_chunk, :]
            rule = self.grow_rule(pos_grow, neg_grow)
            if not rule:
                return rule_set

            if self.prun_ratio > 0:
                pos_prun = pos.iloc[pos_chunk:, :]
                neg_prun = neg.iloc[neg_chunk:, :]
                rule = self.prun_rule(pos_prun, neg_prun, rule)

            rule_dl = self.dl(rule)
            if min_dl + self.dl_threshold < rule_dl:
                return rule_set
            else:
                rule_set.append(rule)
                if rule_dl < min_dl:
                    min_dl = rule_dl

                pos = self.remove_cases(pos, [rule])
                neg = self.remove_cases(neg, [rule])
        return rule_set

    def foil(self, pos, neg, condition, rule=None, ruleset=None):
        if ruleset is None:
            ruleset = []
        if rule is None:
            rule = {}
        ruleset.append(rule)

        if ruleset:
            p0 = np.sum(self.bindings(pos, ruleset))
            n0 = np.sum(self.bindings(neg, ruleset))
        else:
            p0 = len(pos)
            n0 = len(neg)

        ruleset.pop()

        new_rule = dict(rule)
        new_rule[condition[0]] = condition[1]

        ruleset.append(new_rule)

        p1 = np.sum(self.bindings(pos, ruleset))
        n1 = np.sum(self.bindings(neg, ruleset))

        ruleset.pop()

        if p1 < self.sample_threshold:
            return -10000

        if p0 == 0:
            d0 = 0
        else:
            d0 = float(p0) / (float(p0) + float(n0))

        if p1 == 0:
            d1 = 0
        else:
            d1 = float(p1) / (float(p1) + float(n1))

        return math.log(p1, 10) * (d1 - d0 - self.diff_threshold)

    def grow_rule(self, pos, neg, rule=None, ruleset=None):
        if ruleset is None:
            ruleset = []
        if rule is None:
            rule = {}

        pos = self.remove_cases(pos, ruleset)
        neg = self.remove_cases(neg, ruleset)

        while True:
            max_gain = -10000
            max_condition = None

            tmp_conditions = [i for i in self.conditions if i[0] not in rule]
            if len(tmp_conditions) > 0:
                result = Parallel(n_jobs=self.n_jobs)(delayed(self.foil)(pos, neg, condition, rule, ruleset) for condition in tmp_conditions)
                max_gain = max(result)
                max_condition = tmp_conditions[np.argmax(result)]

            self.grow_rule_iter_num += 1
            print('condition:', max_condition, max_gain, rule, self.grow_rule_iter_num)

            if max_gain <= 0:
                return rule

            rule[max_condition[0]] = max_condition[1]
            ruleset.append(rule)

            if np.sum(self.bindings(neg, ruleset)) == 0:
                ruleset.pop()
                return rule

            ruleset.pop()

    def prun_rule(self, pos, neg, rule, ruleset=None):
        if ruleset is None:
            ruleset = []

        # Deep copy our rule.
        tmp_rule = dict(rule)
        # Append the rule to the rules list.
        ruleset.append(tmp_rule)

        p = np.sum(self.bindings(pos, ruleset))
        n = np.sum(self.bindings(neg, ruleset))

        # TODO: 无效rule为何不直接返回空dict{}
        if p == 0 and n == 0:
            ruleset.pop()
            return tmp_rule

        max_rule = dict(tmp_rule)
        max_score = (p - n) / float(p + n)

        keys = list(max_rule.keys())
        i = -1

        while len(tmp_rule.keys()) > 1:
            # Remove the last attribute.
            # 这里的删减是有序的。但是grow过程的condtition学习真的可以保证先学到的比后学到的好么？
            del tmp_rule[keys[i]]

            # Recalculate score.
            p = np.sum(self.bindings(pos, ruleset))
            n = np.sum(self.bindings(neg, ruleset))

            tmp_score = (p - n) / float(p + n)

            # We found a new max score, save rule.
            if tmp_score > max_score:
                max_rule = dict(tmp_rule)
                max_score = tmp_score

            i -= 1

        # Remove the rule from the rules list.
        ruleset.pop()

        return max_rule

    def optimize(self, pos, neg, ruleset):
        new_ruleset = list(ruleset)

        pos_chunk = int((1 - self.prun_ratio) * pos.shape[0])
        neg_chunk = int((1 - self.prun_ratio) * neg.shape[0])

        pos_grow = pos.iloc[:pos_chunk, :]
        neg_grow = neg.iloc[:neg_chunk, :]

        if self.prun_ratio > 0:
            pos_prun = pos.iloc[pos_chunk:, :]
            neg_prun = neg.iloc[neg_chunk:, :]

        i = 0
        while i < len(new_ruleset):
            rule = new_ruleset.pop(i)

            reprule = self.grow_rule(pos_grow, neg_grow)
            if self.prun_ratio > 0:
                reprule = self.prun_rule(pos_prun, neg_prun, reprule, new_ruleset)

            # greedily on whole dataset
            revrule = self.grow_rule(pos, neg, rule, new_ruleset)

            rule_dl = self.dl(rule)
            reprule_dl = self.dl(reprule)
            revrule_dl = self.dl(revrule)

            if (reprule_dl < rule_dl and reprule_dl < revrule_dl):
                # Don't allow duplicates.
                if not reprule in new_ruleset:
                    new_ruleset.insert(i, reprule)
            elif (revrule_dl < rule_dl and revrule_dl < reprule_dl):
                # Don't allow duplicates.
                if not revrule in new_ruleset:
                    new_ruleset.insert(i, revrule)
            else:
                # Don't allow duplicates.
                if not rule in new_ruleset:
                    new_ruleset.insert(i, rule)

            i += 1

        return new_ruleset

    def dl(self, rule):
        """
        Finds the description length for a rule.

        Key arguments:
        rule -- the rule.
        """
        k = len(rule.keys())
        p = k / float(self.init_dl)

        p1 = float(k) * math.log(1 / p, 2)
        p2 = float(self.init_dl - k) * math.log(1 / float(1 - p), 2)

        return int(0.5 * (math.log(k, 2) + p1 + p2))

    def _get_conditions(self, df):
        s = df.dtypes

        # 如果一列在训练集中只出现两个值，而在测试集中出现第三个值，那么不考虑第三个值，我觉得是合理的。毕竟未见的本就不好归类
        binary_cols = []
        for c in df.columns:
            if df[c].nunique() == 2:
                binary_cols.append(c)

        discrete_cols = list(s.index[s == 'object'])
        discrete_cols = [c for c in discrete_cols if c not in binary_cols]

        category_cols = list(s.index[s == 'category'])
        continuous_cols = [i for i in df.columns if i not in discrete_cols + category_cols + binary_cols]

        conditions = []

        for c in binary_cols:
            for v in df[c].unique():
                conditions.append((str(c), str(c) + ' == ' + str(v)))

        for c in discrete_cols:
            for v in df[c].unique():
                conditions.append((str(c), str(c) + ' == ' + '"' + str(v) + '"'))
                conditions.append((str(c), str(c) + ' != ' + '"' + str(v) + '"'))

        for c in continuous_cols:
            _, r = pd.qcut(df[c], q=self.q, retbins=True, duplicates='drop')
            for v in r:
                conditions.append((str(c), str(c) + ' >= ' + str(v)))
                conditions.append((str(c), str(c) + ' <= ' + str(v)))

        for c in category_cols:
            for v in df[c].unique():
                conditions.append((str(c), str(c) + ' >= ' + str(v)))
                conditions.append((str(c), str(c) + ' <= ' + str(v)))

        self.conditions = conditions
        self.init_dl = len(conditions)

    def bindings(self, df, ruleset):
        l_t = pd.Series(np.zeros(df.shape[0]), index=df.index).astype(bool)

        for rule in ruleset:
            l = pd.Series(np.ones(df.shape[0]), index=df.index).astype(bool)
            if len(rule) > 0:
                l &= df.eval(' & '.join(rule.values()))
            l_t |= l
        return np.array(l_t)

    def remove_cases(self, df, ruleset):
        l_t = self.bindings(df, ruleset)
        df = df[~l_t]
        return df

class RipperkClassifier(Ripperk, RulerMixin):
    def fit(self, df, label,feature_names=None, sample_weight=None):
        if feature_names is not None:
            df_copy = pd.DataFrame(df)
            df_copy.columns = feature_names
        else:
            df_copy = df.copy()
        self.feature_names_ = list(df_copy.columns)

        super().fit(df_copy, label)
        self.rules_ = []
        for dic in self.rulesets[1]:
            self.rules_.append((' & '.join(dic.values()), (np.nan, np.nan, np.nan)))

    def predict_proba(self, df):
        for k,v in self.rules_:
            try:
                t += df.eval(k).astype(int)
            except:
                t = df.eval(k).astype(int)
        t /= len(self.rules_)
        return np.concatenate([np.array(t).reshape(t.shape[0], -1), np.array(1-t).reshape(t.shape[0], -1)], axis=1)


if __name__ == '__main__':
    def get_rootpath(path):
        return os.path.join(path.rsplit('/mlearn', 1)[0], 'mlearn')

    root_path = get_rootpath(os.path.abspath(__file__))
    path = get_rootpath(root_path)
    df = pd.read_pickle(os.path.join(path, 'data/xk_v4_data.pkl')).fillna(-999)
    del df['apply_risk_id'], df['apply_risk_created_at']
    y = df.pop('overdue_days')
    rp = Ripperk()
    rp.fit(df, y)
    pred = rp.predict(df)