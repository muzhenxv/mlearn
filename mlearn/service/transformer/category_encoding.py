# from category_encoders import *
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from collections import defaultdict
from .continous_encoding import *
from sklearn.pipeline import Pipeline

class CountEncoder(BaseEstimator, TransformerMixin):
    """
    count encoding: Replace categorical variables with count in the train set.
    replace unseen variables with 1.
    Can use log-transform to be avoid to sensitive to outliers.
    Only provide log-transform with base e, because I think it's enough.


    Attributes
    ----------
    map: a collections.Counter(which like dict) map variable's values to its frequency.

    Example
    -------
    enc = countencoder()
    enc.fit(['a','b','c', 'b', 'c', 'c'])
    enc.transform(['a','c','b'])
    Out:
    array([ 0.        ,  1.09861229,  0.69314718])

    """

    def __init__(self, unseen_value=0, log_transform=True, smoothing=1, inplace=True, prefix='_count'):
        """

        :param unseen_value: 在训练集中没有出现的值给予unseen_value的出现频次，然后参与smoothing
        :param log_transform: 是否取log
        :param smoothing: 光滑处理，在出现频次上+smoothing
        :param inplace: 是否删除原始字段
        """
        self.unseen_value = unseen_value
        self.log_transform = log_transform
        self.smoothing = smoothing
        self.inplace = inplace
        self.map = {}
        self.prefix = prefix

    def fit(self, X, y=None):
        """
        :param X: df
        :param y: None
        :return:
        """
        # TODO: 必须copy。不然使用df[c].replace这样的操作，会直接作用在X上。虽然不加copy，df和X， df[c]和X[c]的内存地址已经不一样。待查明！？
        # 如果不强转为str，那么如果有一列为float型，那么通过Counter或者pd.unique方法得到的结果result虽然也会有nan的存在，但是进行np.nan in result判断时会报False.
        df = pd.DataFrame(X.copy()).astype(str)

        self._set_unseen_key(df)

        for c in df.columns:
            dmap = Counter(df[c])
            for k in dmap.keys():
                dmap[k] += self.smoothing
            dmap[self.unseen_key] = self.unseen_value + self.smoothing
            self.map[c] = dmap

        self.columns = list(df.columns)
        return self

    def transform(self, X):
        """
        :param X: df
        :return:
        """
        # df[c].replace方法要求入参的字典key的类型必须和df[c]本身的数值类型一致，改用map方法无此问题
        df = pd.DataFrame(X.copy()).astype(str)
        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        for c in self.columns:
            l = [i for i in df[c].unique() if i not in self.map[c].keys()]
            if len(l) > 0:
                df[c].replace(l, self.unseen_key, inplace=True)
            df[str(c) + self.prefix] = df[c].map(self.map[c])
            if self.inplace:
                del df[c]
        if self.log_transform:
            X = np.log(df)
        return X

    def _set_unseen_key(self, df):
        if type(df) != pd.DataFrame:
            df = pd.DataFrame(df)
        t = 'unknown'
        while t in df.values:
            t += '*'
        self.unseen_key = t


class CateLabelEncoder(BaseEstimator, TransformerMixin):
    """
    sklearn.preprocess.LabelEncoder can't process values which don't appear in fit label encoder.
    this method can process this problem. Replace all unknown values to a certain value, and encode this
    value to 0.

    Attributes
    ----------
    like sklearn.preprocess.LabelEncoder

    Example
    -------
    enc = labelencoder()
    enc.fit(['a','b','c'])
    enc.transform(['a','v','d'])
    Out: array([1, 0, 0])

    """

    def __init__(self, inplace=True, prefix='_label'):
        self.map = {}
        self.inplace = inplace
        self.prefix = prefix

    def fit(self, X, y=None):
        """
        :param X: array-like of shape (n_samples,)
        :param y: None
        :return:
        """
        df = pd.DataFrame(X.copy()).astype(str)

        self._set_unseen_key(df)

        num_start = 0
        for c in df.columns:
            unique_key = list(df[c].unique())
            unique_num = len(unique_key)
            self.map[c] = dict(zip(unique_key + [self.unseen_key], range(num_start, num_start + unique_num + 1)))
            num_start += unique_num + 1

        self.columns = list(df.columns)
        return self

    def transform(self, X):
        """
        :param X: array-like of shape (n_samples,)
        :return:
        """
        df = pd.DataFrame(X.copy()).astype(str)
        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        for c in self.columns:
            l = [i for i in df[c].unique() if i not in self.map[c].keys()]
            if len(l) > 0:
                df[c].replace(l, self.unseen_key, inplace=True)
            df[str(c) + self.prefix] = df[c].map(self.map[c])
            if self.inplace:
                del df[c]
        return df

    def _set_unseen_key(self, df):
        if type(df) != pd.DataFrame:
            df = pd.DataFrame(df)
        t = 'unknown'
        while t in df.values:
            t += '*'
        self.unseen_key = t


# class CateOneHotEncoder(BaseEstimator, TransformerMixin):
#     def __init__(self, n_values="auto", categorical_features="all",
#                  dtype=np.float64, sparse=True, handle_unknown='ignore', sparse_thr=200):
#         self.n_values = n_values
#         self.categorical_features = categorical_features
#         self.dtype = dtype
#         self.sparse = sparse
#         self.handle_unknown = handle_unknown
#         self.sparse_thr = sparse_thr
#         self.onehot_params = self.get_params()
#         del self.onehot_params['sparse_thr']

#     def fit(self, X, y=None):
#         df = pd.DataFrame(X.copy())
#         self.le = CateLabelEncoder()
#         self.le.fit(df)

#         df = self.le.transform(df)

#         self.onehot = OneHotEncoder(**self.onehot_params)
#         self.onehot.fit(df)
#         return self

#     def transform(self, X):
#         df = pd.DataFrame(X.copy())
#         df = self.le.transform(df)
#         df = self.onehot.transform(df)
#         if df.shape[1] < self.sparse_thr:
#             df = pd.DataFrame(df.todense())
#             df.columns = [str(i) + '_onehot' for i in df.columns]
#         return df


class CateOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_values="auto", categorical_features="all",
                 dtype=np.float64, sparse=True, handle_unknown='ignore', sparse_thr=200):
        self.n_values = n_values
        self.categorical_features = categorical_features
        self.dtype = dtype
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.sparse_thr = sparse_thr
        self.onehot_params = self.get_params()
        del self.onehot_params['sparse_thr']

    def fit(self, X, y=None):
        df = pd.DataFrame(X.copy())
        self.encoder = Pipeline([('catlabelencoder', CateLabelEncoder()), ('onehotencoder', OneHotEncoder(**self.onehot_params))])
        self.encoder.fit(df)
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        df = self.encoder.transform(df)
        if not self.sparse:
            df = pd.DataFrame(df.todense())
            df.columns = [str(i) + '_onehot' for i in df.columns]
        return df


def _woe_iv(x, y, woe_min=-20, woe_max=20, nan_woe=None, limit=True):
    # TODO: woe_min&woe_max设置是否合理？
    """

    :param x: array
    :param y: array
    :return:
    """
    x = pd.Series(x)
    unique_k = list(np.unique(x.dropna()))
    missing = 'missing'
    while missing in x:
        missing += 'missing'

    x = np.array(x.fillna(missing))
    y = np.array(y)

    if missing in x:
        unique_k.append(missing)

    pos = (y == 1).sum()
    neg = (y == 0).sum()

    dmap = {}

    iv = 0
    for k in unique_k:
        if (k == missing) & (nan_woe is not None):
            woe1 = nan_woe
        else:
            indice = np.where(x == k)
            pos_r = (y[indice] == 1).sum() / pos
            neg_r = (y[indice] == 0).sum() / neg

            if (pos_r == 0) & (neg_r == 0):
                woe1 = 0
            elif (pos_r == 0) | (pos_r is np.nan):
                woe1 = woe_min
            elif (neg_r == 0) | (neg_r is np.nan):
                woe1 = woe_max
            else:
                woe1 = math.log(pos_r / neg_r)
                if limit:
                    woe1 = min(woe1, woe_max)
                    woe1 = max(woe1, woe_min)

        dmap[k] = woe1
        iv += (pos_r - neg_r) * woe1
    if len(unique_k) == 1:
        iv = np.nan
    return dmap, iv


class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    woe变换

    适用于cont和cate，但对多取值cont无效，支持缺失值

    Parameters
    ----------
    diff_thr : int, default: 20
        不同取值数小于等于该值的才进行woe变换，不然原样返回

    woe_min : int, default: -20
        woe的截断最小值

    woe_max : int, default: 20
        woe的截断最大值

    nan_thr : float, default: 0.01
        对缺失值采用平滑方法计算woe值，nan_thr为平滑参数

    """

    def __init__(self, diff_thr=20, woe_min=-20, woe_max=20, nan_thr=0.01, inplace=True, suffix='_woe', limit=True,
                 **kwargs):
        self.diff_thr = diff_thr
        self.woe_min = woe_min
        self.woe_max = woe_max
        self.nan_thr = nan_thr
        self.inplace = inplace
        self.suffix = suffix
        self.limit = limit
        self.map = {}
        self.ivmap = {}

    def fit(self, X, y):
        df = pd.DataFrame(X.copy())
        self.map = {}
        self.columns = list(df.columns)
        t = df.nunique()
        self.woecols = list(t[t <= self.diff_thr].index)
        self.nowoecols = [c for c in self.columns if c not in self.woecols]

        y = pd.DataFrame(y)
        label = 'label'
        y.columns = [label]
        for c in self.woecols:
            tmp = pd.concat([df[c], y], axis=1)

            nan_woe = self.nan_woe_cmpt(tmp, c, label, self.nan_thr, self.woe_min, self.woe_max)

            # 计算woe时总正样例数需要考虑nan情况, 将nan_woe作为缺失值对应的woe值
            # tmp = tmp.dropna()

            dmap, iv = _woe_iv(tmp[c], tmp[label], self.woe_min, self.woe_max, limit=self.limit)
            dmap[np.nan] = nan_woe

            self.map[c] = dmap
            self.ivmap[c] = iv
        # for c in self.nowoecols:
        #     self.map[c] = None
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        for c in self.woecols:
            nan_val = self.map[c][np.nan]
            df[c + self.suffix] = df[c].map(lambda x: self.map[c].get(x, nan_val))
            if self.inplace & (self.suffix != ''):
                del df[c]
        return df

    @staticmethod
    def nan_woe_cmpt(df, col, label='label', nan_thr=0.01, woe_min=-20, woe_max=20, limit=True):
        m = df[label].mean()
        pos_base = int(df.shape[0] * m * nan_thr)
        neg_base = int(df.shape[0] * (1 - m) * nan_thr)
        tmp = df[df[col].isnull()]
        t = tmp[label].shape[0]
        pos = tmp[label].sum()
        neg = t - pos
        pos_r = (pos + pos_base) / df[label].sum()
        neg_r = (neg + neg_base) / (df.shape[0] - df[label].sum())

        if (pos_r == 0) & (neg_r == 0):
            woe1 = 0
        elif (pos_r == 0) | (pos_r is np.nan):
            woe1 = woe_min
        elif (neg_r == 0) | (neg_r is np.nan):
            woe1 = woe_max
        else:
            woe1 = math.log(pos_r / neg_r)
            if limit:
                woe1 = min(woe1, woe_max)
                woe1 = max(woe1, woe_min)

        return woe1


class CateBinningEncoder(BaseEstimator, TransformerMixin):
    # TODO: 强依赖ylabel，应该支持不依赖y的归并方式
    """
    对离散值变量做归并

    仅适用于cate， 支持缺失值

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

    Attributes
    ----------
    map: dict 每个特征每个取值所属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'a': 1, 'b': 1, 'c': 2}}
    kmap: dict 每个特征每一类取值组成的list及其归属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'[a, b]': 1, '[c]': 2}}

    """

    def __init__(self, diff_thr=20, bins=10, binning_method='monoc', inplace=True, suffix='_bin', **kwargs):
        self.diff_thr = diff_thr
        self.bins = bins
        self.binning_method = binning_method
        self.inplace = inplace
        self.kwargs = kwargs
        self.suffix = suffix
        self.map = {}
        self.kmap = {}

    def fit(self, X, y=None):
        if y is None:
            if self.binning_method != 'dt':
                y = [np.random.randint(0, 2) for _ in range(len(X))]
            else:
                raise ValueError('y must be not None if binning_method is dt!')

        df = pd.DataFrame(X.copy())

        target = pd.DataFrame(y)
        target.columns = ['label']

        self.columns = list(df.columns)

        t = df.nunique()
        self.bin_cols = list(t[t > self.diff_thr].index)
        self.notbin_cols = list(t[t <= self.diff_thr].index)

        self.map = defaultdict(dict)
        self.kmap = defaultdict(dict)
        woe_enc = WOEEncoder(diff_thr=np.inf)
        tmp = woe_enc.fit_transform(df[self.bin_cols], target)
        contbin_enc = ContBinningEncoder(diff_thr=self.diff_thr)
        tmp2 = contbin_enc.fit_transform(tmp, target)
        for c in woe_enc.map.keys():
            if c + '_woe_bin' in tmp2.columns:
                c_b = c + '_woe_bin'
            else:
                c_b = c + '_woe'
            # dp = self.invert_dict_nonunique(woe_enc.map[c])
            for v in tmp2[c_b].unique():
                #                 print(pd.Series(df[c][tmp2[c+'_woe_bin']==v].unique()))
                k = list(pd.Series(df[c][tmp2[c_b] == v].unique()))
                #                 print(l)
                #                 k = [item for sublist in l for item in sublist]
                for ksub in k:
                    self.map[c][ksub] = v
                self.kmap[c][str(k)] = v
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        for c in self.bin_cols:
            df[c + self.suffix] = df[c].map(self.map[c])
            if self.inplace & (self.suffix != ''):
                del df[c]

        return df

    @staticmethod
    def invert_dict_nonunique(d):
        newdict = {}
        for k, v in d.items():
            newdict.setdefault(v, []).append(k)
        return newdict
