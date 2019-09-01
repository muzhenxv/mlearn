import numpy as np
import pandas as pd
from functools import reduce
from sklearn import metrics
from itertools import chain

from .tree_utils import RuleCondition, Rule
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class Tree2Rule():
    def __init__(self, estimator, type_of_tree='RF', n_estimators=20):
        self.type_of_tree = type_of_tree
        self.n_estimators = n_estimators
        self.clf = estimator

        if not isinstance(self.clf, RandomForestClassifier) and not isinstance(self.clf, GradientBoostingClassifier):
            raise ValueError('INFO : Tree2Rule only works with RandomForestClassifier and GradientBoostingClassifier.')

    def fit(self, X, y):
        self.feature_names = X.columns.tolist()
        self.clf.fit(X.as_matrix(), y.as_matrix())
        if isinstance(self.clf, RandomForestClassifier):
            rules_conditions = [self.tree2rules(singe_tree, self.feature_names) for singe_tree in self.clf.estimators_]
        elif isinstance(self.clf, GradientBoostingClassifier):
            rules_conditions = [self.tree2rules(singe_tree[0], self.feature_names) for singe_tree in
                                self.clf.estimators_]
        else:
            raise ValueError('INFO : Tree2Rule only works with RandomForestClassifier and GradientBoostingClassifier.')

        self.rules_ensemble = [tmp[0] for tmp in rules_conditions]
        self.rules_ensemble = list(chain(*self.rules_ensemble))

        self.conditions_ensemble = [tmp[1] for tmp in rules_conditions]
        self.conditions_ensemble = list(chain(*self.conditions_ensemble))

    def _fit(self, X, y):
        pass

    def transform(self, X):
        """
        # TODO : 根据tree_ensamble转成相关的矩阵
        """
        pass

    def predict(self, X):
        """
        根据transform预测相关结果
        0 / 1
        """
        pass

    def predict_prob(self, X):
        """
        # TODO: 根据transform预测相关概率

        """
        pass

    def get_metrics(self, X, y, split=True, average='weighted'):
        result = []
        # tmp = self.rules_ensemble if cluster else self.conditions_ensemble
        tmp = self.conditions_ensemble if split else self.rules_ensemble
        for i, rule in enumerate(tmp):
            y_pred = rule.transform(X)
            describe = rule.get_str()
            profile = self._get_metrics(y, y_pred, average=average, describe=describe)
            result.append(profile)
        desc = pd.DataFrame(result, columns=['Lift', 'Precision', 'Recall', 'F1_score', 'Hit', 'Cnt', 'Describe'])
        return desc.drop_duplicates()

    def _get_metrics(self, y_true, y_pred, average='weighted', describe=''):
        from sklearn import metrics
        l = self.get_lift(y_true, y_pred, average=average)
        p = metrics.precision_score(y_true, y_pred, average=average)
        r = metrics.recall_score(y_true, y_pred, average=average)
        f1 = metrics.f1_score(y_true, y_pred, average=average)
        c = y_pred.sum()
        h = (y_true & y_pred).sum()
        return [l, p, r, f1, h, c, describe]

    @staticmethod
    def get_lift(y_true, y_pred, average='weighted'):
        """
        TODO: 考虑分母为0的情况
        """
        try:
            lift = ((y_true & y_pred).sum() / y_pred.sum()) / y_true.mean()
        except:
            lift = np.nan
        return lift

    @staticmethod
    def tree2rules(tree, feature_name=None):
        from sklearn.tree import _tree
        tree_ = tree.tree_
        rules = set()
        all_conditions = set()

        def traverse_tree(node_id=0, operator=None, threshold=None, name=None, conditions=[]):
            if node_id == 0:
                tree_rules = []
            else:
                # feature 是 id，需要转换成名字
                node_condition = ' '.join([str(name), operator, str(threshold)])

                # 根据给定的条件，转换成condition的对象
                node_condition = RuleCondition(feature_index=name,
                                               threshold=threshold,
                                               operator=operator,
                                               support=tree_.n_node_samples[node_id] / float(tree_.n_node_samples[0]),
                                               feature_name=name)
                all_conditions.update([node_condition])
                tree_rules = conditions + [node_condition]

            # 非叶子节点
            # 左边节点是小于等于，右边节点是大于
            if tree_.children_left[node_id] != _tree.TREE_LEAF:
                # GET THE CURRENT FEATURE NAME AND THRESHOLD
                name = feature_name[tree_.feature[node_id]]
                threshold = tree_.threshold[node_id]

                # LEFT NODE
                left_node_id = tree_.children_left[node_id]
                traverse_tree(left_node_id, "<=", threshold, name, tree_rules)

                # RIGHT NODE
                right_node_id = tree_.children_right[node_id]
                traverse_tree(right_node_id, ">", threshold, name, tree_rules)

            # 叶子节点
            else:
                if len(tree_rules) > 0:
                    # 将 condiontons 的list转换成Rule格式的对象
                    new_rule = Rule(tree_rules, tree_.value[node_id][0][0])
                    rules.update([new_rule])
                else:
                    pass  # tree only has a root node!
                return None

        traverse_tree()
        return rules, all_conditions



