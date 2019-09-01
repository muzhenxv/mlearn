import pandas as pd
from sklearn.feature_selection import *
from ..data_service import get_data
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import *
from sklearn.tree import *
from sklearn.ensemble import *
from xgboost.sklearn import XGBClassifier
from .base_filter import *
from ..base_utils import instantiate_utils as instantiate


class FeatureFilter(BaseEstimator, TransformerMixin):
    """
    filter
    """

    def __init__(self, st):
        self.st = st

    def fit(self, df_src, label=None, test_src=None, test_label=None):
        self.label = label
        df = get_data(df_src)
        label = df.pop(label)
        if test_src is not None:
            test_src = get_data(test_src)
            if type(test_label) == str:
                if test_label in test_src.columns:
                    test_label = test_src.pop(test_label)
                else:
                    raise ValueError('%s not in test data!' % test_label)

        self.st = self._reset_st(self.st, df.shape[1])
        self.columns = list(df.columns)

        tmp = df
        test_tmp = test_src
        for encoder_dict in self.st:
            enc = instantiate.instantiate(encoder_dict.method, encoder_dict.params)
            tmp = enc.fit_transform(tmp, label, test_tmp, test_label)
            test_tmp = enc.transform(test_tmp)
            encoder_dict['enc'] = enc

            if tmp.shape[1] == 0:
                break
        return self

    def fit_transform(self, df_src, label=None, test_src=None, test_label=None):
        self.label = label
        df = get_data(df_src)
        label = df.pop(label)
        if test_src is not None:
            test_src = get_data(test_src)
            if type(test_label) == str:
                if test_label in test_src.columns:
                    test_label = test_src.pop(test_label)
                else:
                    raise ValueError('%s not in test data!' % test_label)

        self.st = self._reset_st(self.st, df.shape[1])
        self.columns = list(df.columns)

        tmp = df
        test_tmp = test_src
        for encoder_dict in self.st:
            enc = instantiate.instantiate(encoder_dict['method'], encoder_dict['params'])
            tmp = enc.fit_transform(tmp, label, test_tmp, test_label)
            test_tmp = enc.transform(test_tmp)
            encoder_dict['enc'] = enc

            if tmp.shape[1] == 0:
                break

        df_final = pd.concat([tmp, label], axis=1)
        return df_final

    def transform(self, df_src):
        df = get_data(df_src)
        try:
            df_label = df.pop(self.label)
        except:
            df_label = pd.DataFrame()

        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        tmp = df
        for encoder_dict in self.st:
            if 'enc' in encoder_dict:
                tmp = encoder_dict['enc'].transform(tmp)
            else:
                break

        df_final = pd.concat([tmp, df_label], axis=1)
        return df_final

    def _reset_st(self, st, col_num):
        for encoder_dict in st:
            if encoder_dict['method'] == 'SelectKBestFilter':
                if encoder_dict['params']['k'] > col_num:
                    encoder_dict['params']['k'] = 'all'
        return st
