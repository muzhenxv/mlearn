# from category_encoders import *
import numpy as np
from collections import Counter
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from collections import defaultdict
from .continous_encoding import *
from sklearn.pipeline import Pipeline
from ..base_utils.base_utils import infer_dtypes

class CountEncoder(BaseEstimator, TransformerMixin):
    """
    count encoding: Replace categorical variables with frequency in the train set.
    replace unseen variables with 0.


    Attributes
    ----------
    map: a collections.Counter(which like dict) map variable's values to its frequency.

    Example
    -------
    enc = CountEncoder()
    enc.fit(['a','b','c', 'b', 'c', 'c'])
    enc.transform(['a','c','b'])
    Out:
    array([ 0.        ,  1.09861229,  0.69314718])

    """

    def __init__(self, inplace=True, suffix='_count', category_features='infer', cate_threshold=20):
        """
        类别变量按照出现频率编码
        :param inplace: 是否删除原始字段
        :suffix: 新字段后缀
        :category_features: "all", "infer" or array
            Specify what features are treated as categorical.

            - 'infer'(default): 根据字段类型判断，object，category作为处理对象
            - 'all': All features are treated as categorical.
            - array: Array of categorical feature names.

            Non-categorical features are always stacked to the right of the matrix.
        """
        if suffix == '' and inplace == False:
            raise ValueError('suffix is black conflicts with inplace is False!')
        self.inplace = inplace
        self.suffix = suffix
        self.category_features = category_features
        self.cate_threshold = 20

    def fit(self, X, y=None):
        """
        :param X: df
        :param y: None
        :return:
        """
        self.map = {}
        df = pd.DataFrame(X.copy())
        
        if self.category_features == 'infer':
            self.category_columns = infer_dtypes(df, self.cate_threshold, dtype='cate')
        elif self.category_features == 'all':
            self.category_columns = list(df.columns)
        else:
            self.category_columns = self.category_features
            
        for c in self.category_columns:
            self.map[c] = df[c].value_counts(normalize=True, dropna=False).to_dict()

        self.columns = list(df.columns)
        return self

    def transform(self, X):
        """
        :param X: df
        :return:
        """
        df = pd.DataFrame(X.copy())
        assert list(df.columns) == self.columns

        for c in self.category_columns:
            df[str(c) + self.suffix] = df[c].map(self.map[c]).astype(float).fillna(0)
            if self.inplace & (self.suffix != ''):
                del df[c]
        return df


class WOEEncoder(BaseEstimator, TransformerMixin):
    """
    woe变换

    适用于cont和cate，但对多取值cont无效，支持缺失值

    Parameters
    ----------
    diff_thr : int, default: 20
        不同取值数小于等于该值的才进行woe变换，不然原样返回

    woe_min : int, default: -np.log(100)
        woe的截断最小值

    woe_max : int, default: np.log(100)
        woe的截断最大值

    nan_thr : float, default: 0.01
        对缺失值采用平滑方法计算woe值，nan_thr为平滑参数

    """

    def __init__(self, cate_threshold=20, category_features='infer', woe_min=-4.6, woe_max=4.6, inplace=True, suffix='_woe'):
        if suffix == '' and inplace == False:
            raise ValueError('suffix is black conflicts with inplace is False!')
            
        self.cate_threshold = cate_threshold
        self.woe_min = woe_min
        self.woe_max = woe_max
        self.inplace = inplace
        self.suffix = suffix
        self.category_features = category_features
        
    def fit(self, X, y):
        self.map = {}
        self.ivmap = {}   
        
        df = pd.DataFrame(X.copy())
        self.columns = list(df.columns)

        if self.category_features == 'infer':
            self.category_columns = infer_dtypes(df, self.cate_threshold, dtype='cate')
        elif self.category_features == 'all':
            self.category_columns = list(df.columns)
        else:
            self.category_columns = self.category_features        

        target = pd.Series(y, index=df.index)
        for c in self.category_columns:
            pos = df[c][target==1].value_counts(normalize=True, dropna=False)
            neg = df[c][target==0].value_counts(normalize=True, dropna=False)
            tmp = np.log(pos.divide(neg,fill_value=0)).replace([np.inf, -np.inf], [self.woe_max, self.woe_min])
            self.map[c] = tmp.to_dict()
            self.ivmap[c] = np.sum(pos.sub(neg, fill_value=0) * tmp)
            if np.nan not in self.map[c]:
                self.map[c][np.nan] = 0            
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        assert list(df.columns) == self.columns

        for c in self.category_columns:
            df[c + self.suffix] = df[c].map(self.map[c]).astype(float).fillna(self.map[c][np.nan])
            if self.inplace & (self.suffix != ''):
                del df[c]
        return df


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
