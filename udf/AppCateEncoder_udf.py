from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import pickle


class AppCateEncoder_udf(BaseEstimator, TransformerMixin):
    def __init__(self, cate_dict, delimiter=',', prefix='app_', unknown='unknown'):
        self.delimiter = delimiter
        self.prefix = prefix
        self.unknown = unknown

        if type(cate_dict) == str:
            cate_dict = pickle.load(open(cate_dict, 'rb'))
        self.cate_list = [self.prefix + i for i in set(cate_dict.values()) | set([self.unknown])]
        self.cate_dict = cate_dict

    def fit(self, df, y=None):
        self.columns = self.cate_list + [i + '_ratio' for i in self.cate_list]

        return self

    def transform(self, df):
        dfs = df.squeeze().map(lambda s: self.app2cate(s, self.cate_dict, self.delimiter, self.prefix, self.unknown))
        df = dfs.apply(pd.Series)

        for c in self.cate_list:
            if c not in df.columns:
                df[c] = np.nan
        df[df.notnull().any(axis=1)] = df[df.notnull().any(axis=1)].replace(np.nan, 0)

        tmp = df.divide(df.sum(axis=1), axis=0)
        tmp.columns = [i + '_ratio' for i in df.columns]
        df = pd.concat([df, tmp], axis=1)
        df = df[self.columns]
        return df

    @staticmethod
    def app2cate(x, cate_dict, delimiter=',', prefix='app_', unknown='unknown'):
        try:
            x = set([s.strip() for s in x.split(delimiter)])
            l = [prefix + cate_dict.get(i, unknown) for i in x]
            dic = dict(Counter(l))
        except:
            dic = {}

        return dic


if __name__ == '__main__':
    import os
    path = os.getcwd()
    subpath = path.split('mlearn', 1)[0] + 'mlearn/'
    if os.path.exists(subpath + 'mlearn'):
        cate_dict = subpath + 'mlearn/' + 'materials/cate_dict_20180806_v1.pkl'
    else:
        cate_dict = subpath + 'mlearndev/' + 'materials/cate_dict_20180806_v1.pkl'
    df_online = pd.DataFrame(
        {'equipment_app_name': [np.nan, '分期乐,小赢卡贷,大圣钱包', '汽车之家,Google服务框架', '五行云,腾讯地图,起点日历,云付,百度网盘']})
    enc = AppCateEncoder_udf(cate_dict=cate_dict)
    df = enc.fit_transform(df_online.equipment_app_name)
    print(df)