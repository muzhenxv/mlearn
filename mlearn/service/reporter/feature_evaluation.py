import math
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn import metrics as mr
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

from . import feature_encoding
from ..transformer.category_encoding import *
from ..transformer.continous_encoding import *
from ..transformer.base_encoding import *
import copy
from ..base_utils import *


def count_binary(a, event=1):
    """
    count the number of a's values
    :param a:
    :param event:
    :return:
    """
    event_count = (a == event).sum()
    non_event_count = a.shape[-1] - event_count
    return event_count, non_event_count


def woe_iv(x, y, event=1, limit=True, **kwargs):
    """
    calculate woe and information for a single feature
    :param x: 1-D numpy stands for single feature
    :param y: 1-D numpy array target variable
    :param event: value of binary stands for the event to predict
    :return: dictionary contains woe values for categories of this feature
             information value of this feature
    """
    x = np.array(x)
    y = np.array(y)

    enc = feature_encoding.binningencoder(**kwargs)
    enc.fit(x, y)
    x = enc.transform(x)

    event_total, non_event_total = count_binary(y, event=event)
    x_labels = np.unique(x)
    woe_dict = {}
    iv = 0
    for x1 in x_labels:
        # this for array,如果传入的是pd.Series，y会按照index切片，但是np.where输出的却是自然顺序，会出错
        y1 = y[np.where(x == x1)[0]]
        event_count, non_event_count = count_binary(y1, event=event)
        rate_event = 1.0 * event_count / event_total
        rate_non_event = 1.0 * non_event_count / non_event_total
        if rate_event == 0:
            woe1 = -20
        elif rate_non_event == 0:
            woe1 = 20
        else:
            woe1 = math.log(rate_event / rate_non_event)
            if limit:
                woe1 = max(woe1, -20)
                woe1 = min(woe1, 20)
        woe_dict[x1] = woe1
        iv += (rate_event - rate_non_event) * woe1
    return woe_dict, iv


def iv_df(df, labels, columns=None, **kwargs):
    """
    compute iv for every column in columns with every label in labels
    :param df: dataframe
    :param labels: list
    :param columns: list, if None, all features except for labels will be computed iv with every label in labels.
    :return: dataframe
    """
    dic = defaultdict(dict)

    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    for t in labels:
        for c in columns:
            try:
                iv_v = woe_iv(df[c], df[t], **kwargs)[1]
            except:
                iv_v = np.nan
            dic[t][c] = iv_v
    df = pd.DataFrame(dic)
    df.columns = [['IV'] * df.shape[1], df.columns]
    return df


def ks_df(df, labels, columns=None):
    """
    compute ks for every column in columns with every label in labels using scipy.stats.ks_2samp
    :param df:
    :param labels: list
    :param columns:
    :return:
    """
    dic = defaultdict(dict)

    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    for t in labels:
        for c in columns:
            try:
                tmp = pd.to_numeric(df[c])
                ks_v = ks_2samp(tmp[df[t] == 1], tmp[df[t] == 0])[0]
            except:
                ks_v = np.nan
            dic[t][c] = ks_v
    df = pd.DataFrame(dic)
    df.columns = [['KS'] * df.shape[1], df.columns]
    return df


def dt_auc(y, x, n_split=5, max_depth=5, min_samples_leaf=1, max_leaf_nodes=None, random_state=7):
    """
    compute auc for single feature with label by decision tree.
    :param y: array, target values
    :param x: array, single feature values
    :param n_split: int, the number of folds for cross validation.
    :param max_depth:
    :param min_samples_leaf:
    :param max_leaf_nodes:
    :return: float, auc
    """
    x, y = np.array(x).reshape(-1, 1), np.array(y)

    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                random_state=random_state)

    prob = []
    y_true = []
    skf = StratifiedKFold(n_split, shuffle=True)
    for tr_index, te_index in skf.split(x, y):
        dt.fit(x[tr_index], y[tr_index])
        prob.extend(list(dt.predict_proba(x[te_index])[:, 1]))
        y_true.extend(list(y[te_index]))
    auc1 = mr.roc_auc_score(y_true, prob)
    if auc1 < 0.5:
        auc1 = 1 - auc1

    # 对于单变量，尤其是分数类型的单变量，直接计算auc会更加准确，因为通过dt去训练会丢失信息
    auc2 = mr.roc_auc_score(y, x)
    if auc2 < 0.5:
        auc2 = 1 - auc2
    if auc1 < auc2:
        auc = auc2
    else:
        auc = auc1
    return auc


def auc_df(df, labels, columns=None, n_split=5, max_depth=5, min_samples_leaf=1, max_leaf_nodes=None, random_state=7):
    """
    compute auc for every column in columns with every label in labels
    :param df:
    :param labels: list
    :param columns:
    :return:
    """
    dic = defaultdict(dict)

    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    for t in labels:
        for c in columns:
            try:
                tmp = pd.to_numeric(df[c])
                auc = dt_auc(df[t], tmp.fillna(-999), n_split=n_split, max_depth=max_depth,
                             min_samples_leaf=min_samples_leaf,
                             max_leaf_nodes=max_leaf_nodes, random_state=random_state)
                if auc < 0.5:
                    auc = 1 - auc
            except:
                auc = np.nan
            dic[t][c] = auc

    df = pd.DataFrame(dic)
    df.columns = [['AUC'] * df.shape[1], df.columns]
    return df


def feature_evaluation_single(df, labels, columns=None, **kwargs):
    """
    compute auc, iv, ks for every column in columns with every label in labels
    :param df:
    :param labels: list
    :param columns:
    :return:
    """
    if columns is None:
        columns = [c for c in df.columns if c not in labels]

    df_ks = ks_df(df, labels, columns)
    df_auc = auc_df(df, labels, columns)
    df_iv = iv_df(df, labels, columns, **kwargs)
    df = pd.concat([df_auc, df_iv, df_ks], axis=1)
    df = df.sort_index(axis=1)
    return df


def feature_evaluation(df, labels, hue=None, columns=None, **kwargs):
    """
    compute auc, iv, ks for every column in columns with every label in labels group by hue.
    :param df:
    :param labels: list
    :param columns:
    :param hue: str
    :return:
    """
    if columns is None:
        columns = [c for c in df.columns if c not in labels and c != hue]

    res = pd.DataFrame()
    if hue is None:
        return feature_evaluation_single(df, labels, columns=columns, **kwargs)
    s = sorted(df[hue].unique())
    for i in s:
        temp = feature_evaluation_single(df[df[hue] == i], labels, columns=columns, **kwargs)
        temp.columns = pd.MultiIndex.from_product([temp.columns.levels[0], temp.columns.levels[1], [i]])
        res = pd.concat([res, temp], axis=1)
    res = res.sort_index(axis=1)
    return res


class WOEReport(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.missmap = {}

    def fit(self, X, y, **kwargs):
        df = X.copy()
        y_data = copy.copy(y)
        if type(y_data) == str:
            y_data = df.pop(y_data)
        self.enc = BinningEncoder(suffix='', **kwargs)
        tmp = self.enc.fit_transform(df, y_data)

        for c in tmp.columns:
            missing = 'missing'
            while missing in tmp[c].values:
                missing += 'missing'
            self.missmap[c] = missing
        return self

    def transform(self, X, y):
        df = X.copy()
        y_data = copy.copy(y)
        if type(y_data) == str:
            y_data = df.pop(y_data)
        tmp = self.enc.transform(df)
        woe_enc = WOEEncoder(diff_thr=np.inf, suffix='')
        tmp2 = woe_enc.fit_transform(tmp, y_data)

        t = pd.DataFrame()
        for c in tmp.columns:
            t = pd.concat([t, self._single_feature_woe_report(tmp[c], y_data, self.enc.kmap.get(c), woe_enc.map[c],
                                                              woe_enc.ivmap[c])], axis=0)

        return t

    def fit_transform(self, X, y, **kwargs):
        return self.fit(X, y, **kwargs).transform(X, y)

    def _single_feature_woe_report(self, dfs, y, bin_dict, woe_dict, iv):
        missing = self.missmap[dfs.name]
        tmp3 = pd.concat([dfs, y], axis=1)
        t1 = pd.pivot_table(tmp3.fillna(missing), values=y.name, index=dfs.name, aggfunc=[np.mean, len, sum],
                            margins=True)
        t1.columns = ['overdue_ratio', 'total_sample_num', 'overdue_sample_num']
        t1['group_sample_ratio'] = t1['total_sample_num'] / t1.loc['All', 'total_sample_num']

        # All这一汇总行有点多余
        # t1.loc['All', 'woe'] = np.nan
        t1 = t1.drop(['All'], axis=0)

        try:
            t1 = t1.sort_index()
        except:
            pass

        t1['bin'] = t1.index
        # 下面的map会把bin_dict字典中不存在的值全部变成nan，但是对于没有binning过程也就是字典为空的特征，需要手动replace
        t1['bin'].replace([missing, 'All'], np.nan, inplace=True)
        if bin_dict is not None:
            t1['bin'] = t1['bin'].map(self.invert_dict(bin_dict))
        t1['woe'] = pd.to_numeric(t1.index, errors='ignore')
        t1['woe'].replace(woe_dict, inplace=True)

        try:
            t1.loc[missing, 'woe'] = woe_dict[np.nan]
        except:
            pass

        t1['feature_name'] = dfs.name
        t1['level'] = t1.index
        t1.insert(0, 'IV', iv)
        t1['feature_name'] = t1['feature_name'].astype(str)
        t1['level'] = t1['level'].astype(str)
        t1['bin'] = t1['bin'].astype(str)
        t1 = t1.set_index(['feature_name', 'level', 'bin'])
        return t1

    @staticmethod
    def invert_dict(d):
        return dict([(v, k) for k, v in d.items()])


class ResultReport:
    def __init__(self, bins=10, precision=8, sort_f=False, report_dst=None):
        self.bins = bins
        self.precision = precision
        self.sort_f = sort_f
        # self.report_dst = report_dst

    def fit(self, y_pred):
        y_pred_copy = pd.Series(y_pred)
        qcut, ret = pd.qcut(y_pred_copy, q=self.bins, duplicates='drop', retbins=True, precision=self.precision)
        ncut = len(qcut.cat.categories)
        while ncut != qcut.nunique():
            qcut, ret = pd.qcut(y_pred_copy, q=ncut - 1, duplicates='drop', retbins=True, precision=self.precision)
            ncut = len(qcut.cat.categories)
        self.bins = ncut
        self.ret = ret
        self.ret[0] = -np.inf
        self.ret[-1] = np.inf

    def transform(self, dic=None):
        dic_result = {}
        for k, v in dic.items():
            df = pd.DataFrame(v)
            dic_result[k] = self._gen_table(df, self.ret)
        df = merge_multi_df(dic_result, sort_f=self.sort_f)
        # 方法本身不参与存储，避免耦合
        # if self.report_dst is not None:
        #     df.to_pickle(os.path.join(self.report_dst, 'level_report.report'))
        return df

    @staticmethod
    def _gen_table(df, ret):
        df['level'] = pd.cut(df['y_pred'], bins=ret, labels=range(1, len(ret)))
        t = df.groupby('level')['y_true'].agg([len, np.sum, np.mean])

        t['accbad'] = t['sum'].cumsum()
        t['accbadratio'] = t['accbad'] / t['sum'].sum()
        t['accrandombadratio'] = t['len'].cumsum() / t['len'].sum()
        t['badratio'] = t['sum'] / t['sum'].sum()
        t['accoverdue_ratio'] = t['accbad'] / t['len'].sum()
        t['randombadratio'] = t['len'] / t['len'].sum()
        t2 = (t.sort_index(ascending=False)['sum'].cumsum() / t['sum'].sum()) / (
            t.sort_index(ascending=False)['len'].cumsum() / t['len'].sum())
        t2.name = 'lift'

        t = pd.concat([t, df.groupby('level')['y_pred'].agg(np.mean), t2], axis=1)

        t = t[['len', 'randombadratio', 'accrandombadratio', 'y_pred', 'sum', 'badratio', 'accbad',
               'accbadratio', 'mean', 'accoverdue_ratio', 'lift']]
        t.columns = ['样本量', '样本量占比', '累积样本量占比', '平均模型分', '违约样本量', '违约样本占比', '累积违约样本量', '累积违约样本占比', '违约率',
                     '累积违约率', 'lift']
        t.index.name = '信用等级'
        t.index = t.index.astype(int)
        return t


if __name__ == '__main__':
    try:
        df = pd.read_pickle('utils/iv_test.pkl')
    except:
        df = pd.read_pickle('iv_test.pkl')
    t1 = feature_evaluation(df, labels=['y'], max_depth=3)
    t2 = feature_evaluation(df, labels=['y'], max_depth=4)
    t3 = feature_evaluation(df, labels=['y'], max_depth=5)
    t4 = feature_evaluation(df, labels=['y'], binning_method='qcut', bins=5)
    t5 = feature_evaluation(df, labels=['y'], binning_method='qcut', bins=10)
    t6 = feature_evaluation(df, labels=['y'], binning_method='qcut', bins=15)
    t7 = feature_evaluation(df, labels=['y'], binning_method='cut', bins=5)
    t8 = feature_evaluation(df, labels=['y'], binning_method='cut', bins=10)
    t9 = feature_evaluation(df, labels=['y'], binning_method='cut', bins=15)
    print(t1, t2, t3, t4, t5, t6, t7, t8, t9)
