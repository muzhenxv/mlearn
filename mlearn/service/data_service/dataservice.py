import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

def get_data(src, index_col=0):
    """
    如果是文件路径，将文件读入内存，暂时约定csv文件必须带index，且在第一列。
    :param src:
    :return:
    """
    if type(src) == pd.DataFrame:
        df = src.copy()
    elif src[-3:] == 'pkl':
        df = pd.read_pickle(src)
    elif src[-3:] == 'csv':
        df = pd.read_csv(src, index_col=index_col)
    else:
        raise Exception(f'{src} must be csv or pkl or pd.DataFrame!')
    return df


class DataParser(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass


    def fit(self, infile, label_col=None, numeric_cols=None, ignore_cols=None, cate_cols=None):
        self.cate_cols = cate_cols if cate_cols is not None else []
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.ignore_cols = ignore_cols if ignore_cols is not None else []

        dfi = get_data(infile)

        self.gen_feat_dict(dfi)
        self.label_col =label_col
        return self

    def transform(self, infile):
        dfi = get_data(infile)

        has_label = self.label_col in dfi.columns

        if has_label:
            y = dfi[self.label_col].values.tolist()
            dfi.drop([self.label_col], axis=1, inplace=True)
        else:
            y = None

        ids = dfi.index.values.tolist()
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)

        # numeric_Xv = dfi[self.numeric_cols].values.tolist()
        # dfi.drop(self.numeric_cols, axis=1, inplace=True)

        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            elif col in self.numeric_cols:
                dfi[col] = self.feat_dict[col]
            else:
                dfi[col] = dfi[col].map(self.feat_dict[col]).fillna(self.feat_dict[col][np.nan])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        return Xi, Xv, y, ids

    def gen_feat_dict(self, df):
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            if col in self.ignore_cols:
                continue
            elif col in self.numeric_cols:
                self.feat_dict[col] = tc
                tc += 1
            else:
                us = list(df[col].unique())
                us.append(np.nan)
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        self.feat_dim = tc
        return self
