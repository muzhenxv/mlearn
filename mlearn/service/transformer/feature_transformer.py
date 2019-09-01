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
from .feature_gen import *
from ..base_utils import instantiate_utils as iu
from scipy.sparse import hstack, csr_matrix
import dill

class FeatureTransformer(BaseEstimator, TransformerMixin):
    """
    transformer
    """
    def __init__(self, st, dst):
        self.st = st
        self.verbose = st.get('verbose', True)
        self.dst = os.path.join(dst, 'train')
        self.test_dst = os.path.join(dst, 'test')
        if not os.path.exists(self.dst):
            os.mkdir(self.dst)
        if not os.path.exists(self.test_dst):
            os.mkdir(self.test_dst)

    def fit(self, df_src, label):
        self.label = label
        df = get_data(df_src)
        df = df.apply(pd.to_numeric, errors='ignore')
        df_label = df.pop(self.label)

        df = self._reset_st(df)
        df = self._convert_cols_type(df)

        if 'custom' in self.st:
            self.encoder_dict = self.st['cate'] + self.st['cont'] + self.st['custom']
        else:
            self.encoder_dict = self.st['cate'] + self.st['cont']

        df_final = pd.DataFrame()
        for j, dic in enumerate(self.encoder_dict):
            cols = dic['cols']
            if len(cols) == 0:
                continue

            encoders = dic['encoders']
            tmp = df[cols]
            for i, encoder in enumerate(encoders):
                print('start----', tmp.columns)
                print(encoder['method'])
                enc = iu.instantiate(encoder['method'], encoder['params'])
                tmp = enc.fit_transform(tmp, df_label)
                dill.dump(enc, open(os.path.join(self.dst, '%s_%s_%s.pkl' % (encoder['method'], j, i)), 'wb'))

                try:
                    tmp.index = df_label.index
                    pd.concat([tmp, df_label], axis=1).to_pickle(os.path.join(self.dst, 'ds_%s_%s.pkl' % (j, i)))
                except:
                    pass

                encoder.update({'enc': enc})

                if tmp.shape[1] == 0:
                    break

            if tmp.shape[1] == 0:
                continue

            try:
                df_final = pd.concat([df_final, tmp], axis=1)
            except:
                df_final = hstack([csr_matrix(df_final), csr_matrix(tmp)])

        if df_final.shape[1] == 0:
            raise Exception('transformer with incorrect params drops all columns!')
        try:
            df_final = pd.concat([df_final, df_label], axis=1)
        except:
            df_final = csr_matrix(hstack([csr_matrix(df_final), csr_matrix(pd.DataFrame(df_label))]))
        return self

    def fit_transform(self, df_src, label):
        self.label = label
        df = get_data(df_src)
        df = df.apply(pd.to_numeric, errors='ignore')
        df_label = df.pop(self.label)

        df = self._reset_st(df)
        df = self._convert_cols_type(df)

        if 'custom' in self.st:
            self.encoder_dict = self.st['cate'] + self.st['cont'] + self.st['custom']
        else:
            self.encoder_dict = self.st['cate'] + self.st['cont']

        df_final = pd.DataFrame()
        for j, dic in enumerate(self.encoder_dict):
            cols = dic['cols']
            if len(cols) == 0:
                continue

            encoders = dic['encoders']
            tmp = df[cols]
            for i, encoder in enumerate(encoders):
                print('start----', tmp.columns)
                print(encoder['method'])
                # enc = eval(encoder['method'])(**encoder['params'])
                enc = iu.instantiate(encoder['method'], encoder['params'])
                tmp = enc.fit_transform(tmp, df_label)

                dill.dump(enc, open(os.path.join(self.dst, '%s_%s_%s.pkl' % (encoder['method'], j, i)), 'wb'))

                try:
                    tmp.index = df_label.index
                    pd.concat([tmp, df_label], axis=1).to_pickle(os.path.join(self.dst, 'ds_%s_%s.pkl' % (j, i)))
                except:
                    pass

                encoder.update({'enc': enc})

                if tmp.shape[1] == 0:
                    print('break-------', tmp.shape[1])
                    break

            if tmp.shape[1] == 0:
                continue

            try:
                df_final = pd.concat([df_final, tmp], axis=1)
            except:
                df_final = hstack([csr_matrix(df_final), csr_matrix(tmp)])

        if df_final.shape[1] == 0:
            raise Exception('transformer with incorrect params drops all columns!')
        try:
            df_final = pd.concat([df_final, df_label], axis=1)
        except:
            df_final = csr_matrix(hstack([csr_matrix(df_final), csr_matrix(pd.DataFrame(df_label))]))
        return df_final

    def transform(self, df_src):
        df = get_data(df_src)
        df = df.apply(pd.to_numeric, errors='ignore')
        try:
            df_label = df.pop(self.label)
        except:
            df_label = pd.DataFrame()

        df = self._convert_cols_type(df)

        df_final = pd.DataFrame()
        for j, dic in enumerate(self.encoder_dict):
            cols = dic['cols']
            encoders = dic['encoders']
            tmp = df[cols]
            for i, encoder in enumerate(encoders):
                if self.verbose:
                    print('start----', tmp.columns)
                    print(encoder['method'])
                if not 'enc' in encoder:
                    break

                enc = encoder['enc']
                tmp = enc.transform(tmp)
                if self.verbose:
                    try:
                        tmp.index = df_label.index
                        pd.concat([tmp, df_label], axis=1).to_pickle(os.path.join(self.test_dst, 'ds_%s_%s.pkl' % (j, i)))
                    except:
                        pass

            if tmp.shape[1] == 0:
                continue
            try:
                df_final = pd.concat([df_final, tmp], axis=1)
            except:
                df_final = hstack([csr_matrix(df_final), csr_matrix(tmp)])

        try:
            df_final = pd.concat([df_final, df_label], axis=1)
        except:
            df_final = csr_matrix(hstack([csr_matrix(df_final), csr_matrix(pd.DataFrame(df_label))]))
        return df_final

    def _convert_cols_type(self, df):
        for dic in self.st['cate']:
            cols = dic['cols']
            if len(cols) == 0:
                continue
            df[cols] = df[cols].astype(str)

        for dic in self.st['cont']:
            cols = dic['cols']
            if len(cols) == 0:
                continue
            # 乘以1是为了把bool转换为int
            df[cols] = df[cols].apply(pd.to_numeric) * 1
        return df

    def _reset_st(self, df):
        if self.st['method'] == 'auto':
            if (len(self.st['cate']) > 1) | (len(self.st['cont']) > 1):
                raise Exception('if type is auto, the length of cate list and cont list must be one!')

            cols = df.nunique()[df.nunique() < self.st['params']['thr']].index.values
            df[cols] = df[cols].astype(str)

            tmp = df.dtypes.map(is_numeric_dtype)
            continous_features = list(tmp[tmp].index.values)
            categorial_features = list(tmp[~tmp].index.values)

            if 'custom' in self.st:
                custom_features = [d['cols'] for d in self.st['custom']]
                custom_features = list(chain.from_iterable(custom_features))
                categorial_features = [i for i in categorial_features if i not in custom_features]
                continous_features = [i for i in continous_features if i not in custom_features]
            self.st['cate'][0]['cols'] = categorial_features
            self.st['cont'][0]['cols'] = continous_features
        return df
