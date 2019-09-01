import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from .continous_encoding import *
from .category_encoding import *
from .custom_encoding import *
from .base_encoding import *
from ..data_service import get_data
import json
import pickle
import os
from itertools import chain
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import *
from sklearn.decomposition import *
import inspect


class ReduceGen(BaseEstimator, TransformerMixin):
    """
    支持cluster， decomposition模块相关方法，已测试方法有kmeans，pca
    """

    def __init__(self, method='KMeans', method_params=None, prefix=None):
        self.method = method
        if method_params is None:
            self.method_params = {'random_state': 7}
        else:
            self.method_params = method_params
            self.method_params['random_state'] = 7
        self.prefix = method if prefix is None else prefix

    def fit(self, df, y=None):
        self.columns = list(df.columns)

        self.enc = eval(self.method)(**self.method_params)
        self.fill_values = df.mean()
        self.enc.fit(df.fillna(self.fill_values))
        return self

    def transform(self, df):
        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')
        tmp = self.enc.transform(df.fillna(self.fill_values))
        tmp = pd.DataFrame(tmp, columns=[self.prefix + '_' + str(i) for i in range(tmp.shape[-1])], index=df.index)
        return pd.concat([df, tmp], axis=1)


def cate_stat(dfs):
    """
    适用于反欺诈场景
    对历史流水情况，针对离散型变量，求当前取值的出现次数和频率，以及和众数相比的次数和频率。
    对于反欺诈而言，每个可能取值的次数和频率以及distinct数貌似作为特征没有太大意义
    :param dfs: pd.Series, 第一笔必须为当前流水
    :return:
    """
    dfs.fillna('missing', inplace=True)
    name = str(dfs.name)
    base = dfs.iloc[0]
    t1 = dfs.value_counts()
    n1 = int(t1.loc[base])
    nmod = int(t1.max())
    nsum = int(t1.sum())
    return json.dumps(
        {name + '_base_count': n1, name + '_base_ratio': n1 / nsum, name + '_mod_base_count_diff': nmod - n1,
         name + '_mod_base_ratio_diff': (nmod - n1) / nsum})


def std(x):
    return np.std(x, ddof=0)


def delta_stat(dfs):
    t = dfs.diff()
    name = str(dfs.name)
    base = dfs.iloc[0]
    t = t - base
    t = t.iloc[1:, ]
    dic = {}
    for func in [np.mean, np.median, np.max, np.min, std]:
        dic[name + '_' + inspect.stack()[0][3] + '_' + func.__name__] = func(t)
    return json.dumps(dic)


def _singlebody_baseRFM(df, uid, cont_variable, cont_func, cate_variable, cate_func, delta_variable, delta_func,
                        span_key):
    t = df.groupby(uid)[cont_variable].agg(cont_func)
    t.columns = ['_'.join(col).strip() for col in t.columns.values]

    cate_t = pd.DataFrame()
    if cate_variable is not None:
        for c in cate_variable:
            t2 = df.sort_values(span_key).groupby(uid)[c].agg(cate_func)
            cate_t = pd.concat([cate_t, t2.iloc[:, 0].map(json.loads).apply(pd.Series)], axis=1)

    delta_t = pd.DataFrame()
    if delta_variable is not None:
        for c in delta_variable:
            t2 = df.sort_values(span_key).groupby(uid)[c].agg(delta_func)
            delta_t = pd.concat([delta_t, t2.iloc[:, 0].map(json.loads).apply(pd.Series)], axis=1)

    df_final = pd.concat([t, cate_t, delta_t], axis=1)
    return df_final


def singlebody_rfmTransformer(df, uid, body='body', cont_variable=None, cont_func=None, cate_variable=None,
                              cate_func=None, delta_variable=None, delta_func=None, span_type=None, span_key=None,
                              span_list=None):
    """
    支持rfm系列特征衍生
    :param df:
    :param uid:
    :param cont_variable:
    :param cont_func:
    :param cate_variable:
    :param cate_func:
    :param span_key:
    :param span_list:
    :return:
    """
    if (cont_variable is None) & (cate_variable is None) & (delta_variable is None):
        raise ValueError('cont_variable & cate_variable & delta_variable must be not all None!')
    elif span_key is None:
        raise ValueError('span key must be not None!')

    if cont_func is None:
        cont_func = [np.mean, np.nanmedian, np.max, np.min, std]
    if delta_func is None:
        delta_func = [delta_stat]
    if cate_func is None:
        cate_func = [cate_stat]

    if span_list is None:
        df_final = _singlebody_baseRFM(df, uid, cont_variable, cont_func, cate_variable, cate_func, delta_variable,
                                       delta_func, span_key)
    else:
        df_final = pd.DataFrame()
        for s in span_list:
            if span_type != 'rank':
                tmp = df[df[span_key] < s].sort_values(span_key)
            else:
                tmp = df.loc[df.groupby(uid)[span_type].nlargest(s).index.levels[1]].sort_values(span_key)
            tmp = _singlebody_baseRFM(tmp, uid, cont_variable, cont_func, cate_variable, cate_func, delta_variable,
                                      delta_func, span_key)
            tmp.columns = ['_'.join([col, 'by', str(span_key), str(s), 'by', str(body)]).strip() for col in
                           tmp.columns.values]
            df_final = pd.concat([df_final, tmp], axis=1)
    return df_final


def rfmTransformer(df, uid, body=None, cont_variable=None, cont_func=None, cate_variable=None, cate_func=None,
                   delta_variable=None, delta_func=None, span_dict=None):
    """

    :param df:
    :param uid: str, 主键，样本id
    :param body: list, must be not None, 主体识别字段列表，取值为0/1
    :param cont_variable: list, 需要聚合的连续型变量
    :param cont_func: list, 连续型变量聚合函数列表
    :param cate_variable: list, 需要聚合的离散型变量
    :param cate_func: list, 离散型变量的聚合函数列表
    :param delta_variable: list, 需要按照特定方法聚合的变量
    :param delta_func: list, delta_variable的聚合方法列表
    :param span_dict: dict, must be not None, like {'span':[('span_1',[1])], 'rank':[('rank_1',[1])]}.
                      只允许span和rank两个key，span类型根据数值进行样本筛选，rank类型取近多少个样本。
                      value为list，每个list为一个元组，元组的第一个值为变量名，第二个值为list，指定以该变量为基准按照这个list轮询进行样本筛选。
    :return:
    """
    df_final = pd.DataFrame()
    for c in body:
        tmp = df[df[c] == 1]
        for k, v in span_dict.items():
            for sub in v:
                df_final = pd.concat([df_final,
                                      singlebody_rfmTransformer(tmp, uid, c, cont_variable, cont_func, cate_variable,
                                                                cate_func, delta_variable, delta_func, k, sub[0],
                                                                sub[1])], axis=1)

    return df_final


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)

    enc = ReduceGen()
    enc.fit(df)
    enc.transform(df)
