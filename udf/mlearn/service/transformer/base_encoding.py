import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from .continous_encoding import *
from .category_encoding import *
from collections import defaultdict

from sklearn.pipeline import Pipeline


def _get_max_same_count(c):
    try:
        return c.value_counts().iloc[0]
    except:
        return len(c)


def _get_same_value_ratio(df):
    t = df.apply(_get_max_same_count) / df.shape[0]
    t.name = 'same_value'
    return t


def _get_missing_value_ratio(df):
    t = df.isnull().mean()
    t.name = 'missing'
    return t


class BaseEncoder(BaseEstimator, TransformerMixin):
    """
    用于剔除缺失值严重列，同值严重列，不同值严重cate列（字符串列如果取值太过于分散，则信息量过低）。

    适用于cont和cate，支持缺失值, 建议放置在encoder序列第一位次

    Parameters
    ----------
    missing_thr: 0.8, 缺失率高于该值的列会被剔除

    same_thr: 0.8, 同值率高于该值的列会被剔除

    caate_thr: 0.9， 取值分散率高于该值的字符串列会被剔除

    Attributes
    ----------
    missing_cols: list, 被剔除的缺失值列

    same_cols: list, 被剔除的同值列

    cate_cols: list, 被剔除的取值分散字符串列

    exclude_cols: list, 被剔除的列名
    """

    def __init__(self, missing_thr=0.8, same_thr=0.8, cate_thr=0.9):
        self.missing_thr = missing_thr
        self.same_thr = same_thr
        self.cate_thr = cate_thr

    def fit(self, X, y=None):
        df = pd.DataFrame(X.copy())
        tmp = df.dtypes.map(is_numeric_dtype)
        categorial_features = tmp[~tmp].index.values

        # 寻找缺失值严重列
        tmp = _get_missing_value_ratio(df)
        self.missing_cols = list(tmp[tmp > self.missing_thr].index.values)

        # 寻找同值严重列
        tmp = _get_same_value_ratio(df)
        self.same_cols = list(tmp[tmp > self.same_thr].index.values)

        # 寻找不同值严重cate列
        if len(categorial_features) > 0:
            tmp = df[categorial_features]
            tmp = tmp.nunique() / df.shape[0]
            self.cate_cols = list(tmp[tmp > self.cate_thr].index.values)
        else:
            self.cate_cols = list([])

        self.exclude_cols = list(set(self.missing_cols + self.same_cols + self.cate_cols))
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        return df.drop(self.exclude_cols, axis=1)


class NothingEncoder(BaseEstimator, TransformerMixin):
    """
    原样返回，不做任何处理。本用于测试，现在transformer支持在encoders序列为空情况下原样返回，此类已无实际用途。

    适用于cont和cate，支持缺失值
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X)


class DropEncoder(BaseEstimator, TransformerMixin):
    """
    此类用于返回空df，换言之删除所有字段。

    适用于cont和cate， 支持缺失值
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame()


class ImputeEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, fillna_value=-999):
        self.fillna_value = fillna_value

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        return df.replace(['nan', np.nan], self.fillna_value)


class StableEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, indice_name='psi', indice_thr=0.2):
        self.indice_name = indice_name
        self.indice_thr = indice_thr

    def fit(self, X, y=None):
        pass

    def transform(self, X):
        pass


class BinningEncoder(BaseEstimator, TransformerMixin):
    """
    将连续型变量转化为离散型

    仅适用于cont， 支持缺失值

    Parameters
    ----------
    diff_thr : int, default: 20
        不同取值数高于该值才进行离散化处理，不然原样返回

    binning_method : str, default: 'dt', {'dt', 'qcut', 'cut'}
        分箱方法, 'dt' which uses decision tree, 'cut' which cuts data by the equal intervals,
        'qcut' which cuts data by the equal quantity. default is 'dt'. if y is None, default auto changes to 'qcut'.

    bins : int, default: 10
        分箱数目， 当binning_method='dt'时，该参数失效

    **kwargs :
        决策树分箱方法使用的决策树参数

    """

    def __init__(self, diff_thr=20, bins=10, binning_method='dt', cate_f=True, inplace=True, suffix='_bin', **kwargs):
        self.diff_thr = diff_thr
        self.bins = bins
        self.binning_method = binning_method
        self.inplace = inplace
        self.kwargs = kwargs
        self.cate_f = cate_f
        self.suffix = suffix
        self.kmap = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X.copy())

        self.columns = list(df.columns)

        # TODO: 特征类型判别规则，不能直接根据类型，太粗糙
        t2 = df.dtypes
        self.cate_cols = list(t2[t2 == object].index)
        self.cont_cols = list(t2[t2 != object].index)

        if (len(self.cont_cols) > 0):
            self.cont_bin_enc = ContBinningEncoder(diff_thr=self.diff_thr, bins=self.bins,
                                                   binning_method=self.binning_method, inplace=self.inplace,
                                                   suffix=self.suffix)
            self.cont_bin_enc.fit(df[self.cont_cols], y)
            self.kmap.update(self.cont_bin_enc.kmap)

        if self.cate_f & (len(self.cate_cols) > 0):
            self.cate_bin_enc = CateBinningEncoder(diff_thr=self.diff_thr, inplace=self.inplace, suffix=self.suffix)
            self.cate_bin_enc.fit(df[self.cate_cols], y)
            self.kmap.update(self.cate_bin_enc.kmap)

        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        df_result = pd.DataFrame()
        if (len(self.cont_cols) > 0):
            df_result = pd.concat([df_result, self.cont_bin_enc.transform(df[self.cont_cols])], axis=1)
        if self.cate_f & (len(self.cate_cols) > 0):
            df_result = pd.concat([df_result, self.cate_bin_enc.transform(df[self.cate_cols])], axis=1)
        return df_result


class BinningWOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, bin_diff_thr=20, bins=10, binning_method='dt', bin_cate_f=True, inplace=True, bin_suffix='_bin',
                 woe_diff_thr=20, woe_min=-20, woe_max=20, woe_nan_thr=0.01, woe_suffix='_woe', woe_limit=True,
                 **kwargs):
        self.bin_diff_thr = bin_diff_thr
        self.bins = bins
        self.binning_method = binning_method
        self.inplace = inplace
        self.kwargs = kwargs
        self.bin_cate_f = bin_cate_f
        self.bin_suffix = bin_suffix

        self.woe_diff_thr = woe_diff_thr
        self.woe_min = woe_min
        self.woe_max = woe_max
        self.woe_nan_thr = woe_nan_thr
        self.woe_suffix = woe_suffix
        self.woe_limit = woe_limit

        self.inplace = inplace

    def fit(self, X, y=None):
        steps = [
            ('Binning', BinningEncoder(diff_thr=self.bin_diff_thr, bins=self.bins, binning_method=self.binning_method,
                                       cate_f=self.bin_cate_f, suffix=self.bin_suffix, inplace=self.inplace)),
            ('WOE', WOEEncoder(diff_thr=self.woe_diff_thr, woe_min=self.woe_min, woe_max=self.woe_max,
                               nan_thr=self.woe_nan_thr, suffix=self.woe_suffix, inplace=self.inplace,
                               limit=self.woe_limit))]
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)


class HistogramEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, diff_thr=20, bins=20, binning_method='qcut'):
        self.bins = bins
        self.diff_thr = diff_thr
        self.binning_method = binning_method
        self.map = {}
        self.result_map = {}

    def fit(self, X, y=None):
        df = X.copy()

        self.cate_cols = list(
            set((df.dtypes.index[df.dtypes == object]) | (df.nunique().index[df.nunique() <= self.diff_thr])))
        self.cont_cols = [c for c in df.columns if c not in self.cate_cols]
        # bool convert to numerical
        df[self.cate_cols] = df[self.cate_cols].astype(str)
        df[self.cont_cols] = df[self.cont_cols] * 1

        for c in self.cont_cols:
            if self.binning_method == 'qcut':
                _, thr = pd.qcut(df[c], q=self.bins, duplicates='drop', retbins=True)
            elif self.binning_method == 'cut':
                _, thr = pd.qcut(df[c], bins=self.bins, retbins=True)

            self.map[c] = sorted(set([-np.inf] + list(thr) + [np.inf]))

        for c in self.cate_cols:
            thr = list(df[c].fillna('histogram_*_missing').value_counts().index[:self.bins])
            self.map[c] = thr
        return self

    def transform(self, X):
        df = X.copy()
        df[self.cate_cols] = df[self.cate_cols].astype(str)
        df[self.cont_cols] = df[self.cont_cols] * 1

        for c in self.cate_cols:
            # 可能存在训练集的取值在验证集完全没有出现的情况
            try:
                dic = df[c].fillna('histogram_*_missing').value_counts().loc[self.map[c]].fillna(0).to_dict()
            except:
                dic = {}

            t = len(df) - sum(dic.values())
            if t > 0:
                dic['histogram_*_others'] = t
            self.result_map[c] = dic

        for c in self.cont_cols:
            tmp = pd.cut(df[c], bins=self.map[c]).value_counts().sort_index()
            tmp.index = tmp.index.astype(str)
            dic = tmp.to_dict()
            t = len(df) - sum(dic.values())
            if t > 0:
                dic['histogram_*_missing'] = t
            self.result_map[c] = dic
        return self.result_map
