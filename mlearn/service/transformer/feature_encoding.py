import numpy as np
import pandas as pd
import bisect
import math
from collections import defaultdict, Counter

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

import datetime
def logger(msg):
    print('[INFO]', datetime.datetime.now(), ':', msg)
                    
def infer_dtypes(df, threshold=5, dtype='cate'):
    """
    dataframe 判断特征类型。首先根据类型判断，然后对数值型做附加判断
    :param df:
    :param threshold: 取值数小于thr，视为离散。
    :param dtype: 'cate' or 'cont'
    :return:
    """
    df = df.apply(pd.to_numeric, errors='ignore')
    cols = df.nunique()[df.nunique() < threshold].index.values
#     df[cols] = df[cols].astype(str)
    cols = list(set(list(df.select_dtypes(exclude=np.number).columns) + list(cols)))

    if dtype == 'cate':
        cols = cols
    elif dtype == 'cont':
        cols = [c for c in df.columns if c not in cols]
    else:
        raise ValueError('param dtype must be assigned as cate or cont!')
    return cols

# TODO: 即使取值数很多，在某几列的占比非常高的情况下，即使要求分箱数为10，最终分箱结果可能依旧只有1，2类。

def dt_cut_points(x, y, max_depth=4, min_samples_leaf=0.05, max_leaf_nodes=None, random_state=7):
    """
    A decision tree method to bin continuous variable to categorical one.
    :param x: The training input samples
    :param y: The target values
    :param max_depth: The maximum depth of the tree
    :param min_samples_leaf: int, float, The minimum number of samples required to be at a leaf node
    :param max_leaf_nodes: Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
    :return: The list of cut points
    """
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes,
                                random_state=random_state)
    dt.fit(np.array(x).reshape(-1, 1), np.array(y))
    th = dt.tree_.threshold
    f = dt.tree_.feature

    # 对于没有参与分裂的节点，dt默认会给-2,所以这里要根据dt.tree_.feature把-2踢掉
    return sorted(th[np.where(f != -2)])

def monot_cut_points(x, y, num_of_bins=10, precision=8):
    # TODO: 该方法通过减少箱数来保证单调性，太粗糙！
    from scipy import stats
    x, y = pd.Series(x), pd.Series(y)
    x_notnull = x[pd.notnull(x)]
    y_notnull = y[pd.notnull(x)]
    r = 0
    while np.abs(r) < 1:
        d1 = pd.DataFrame({"x": x_notnull, "y": y_notnull,
                           "Bucket": pd.qcut(x_notnull, num_of_bins, duplicates='drop')})
        d2 = d1.groupby('Bucket', as_index=True)
        r, p = stats.spearmanr(d2['x'].mean(), d2['y'].mean())
        num_of_bins -= 1
    cp = d2['x'].min().tolist()
    cp.append(d2['x'].max().max())
    return np.round(cp, precision)

def get_cut_points(X, y=None, bins=10, binning_method='dt', precision=8, **kwargs):
    if binning_method == 'cut':
        _, cut_points = pd.cut(X, bins=bins, retbins=True, precision=precision)
    elif binning_method == 'qcut':
        _, cut_points = pd.qcut(X, q=bins, retbins=True, duplicates='drop', precision=precision)
    elif binning_method == 'dt':
        cut_points = dt_cut_points(X, y, **kwargs)
    elif binning_method == 'monot':
        cut_points = monot_cut_points(X, y, precision)
    else:
        raise ValueError("binning_method: '%s' is not defined." % binning_method)

    if binning_method != 'dt':
        cut_points = cut_points[1:-1]

    cut_points = list(cut_points)
    cut_points.append(np.inf)
    cut_points.insert(0, -np.inf)

    return cut_points

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

    def __init__(self, inplace=True, suffix='_count', specified_features='infer', cate_threshold=20):
        """
        类别变量按照出现频率编码
        :param inplace: 是否删除原始字段
        :suffix: 新字段后缀
        :specified_features: "all", "infer" or array
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
        self.specified_features = specified_features
        self.cate_threshold = 20

    def fit(self, X, y=None):
        """
        :param X: df
        :param y: None
        :return:
        """
        self.map = {}
        df = pd.DataFrame(X.copy())
        
        if self.specified_features == 'infer':
            self.specified_columns = infer_dtypes(df, self.cate_threshold, dtype='cate')
        elif self.specified_features == 'all':
            self.specified_columns = list(df.columns)
        else:
            self.specified_columns = self.specified_features    
            
        for c in self.specified_columns:
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

        for c in self.specified_columns:
            df[str(c) + self.suffix] = df[c].map(self.map[c]).astype(float).fillna(0)
            if self.inplace & (self.suffix != ''):
                del df[c]
        return df

class ContBinningEncoder(BaseEstimator, TransformerMixin):
    """
    将连续型变量转化为离散型

    仅适用于cont， 支持缺失值

    Parameters
    ----------
    diff_thr : int, default: 20
        不同取值数高于该值才进行离散化处理，不然原样返回

    binning_method : str, default: 'dt', {'dt', 'qcut', 'cut', 'monot'}
        分箱方法, 'dt' which uses decision tree, 'cut' which cuts data by the equal intervals,
        'qcut' which cuts data by the equal quantity. default is 'dt'. if y is None, default auto changes to 'qcut'.

    bins : int, default: 10
        分箱数目， 当binning_method='dt'时，该参数失效

    **kwargs :
        决策树分箱方法使用的决策树参数

    Attributes
    ----------
    map: dict 每个特征每个取值所属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'cut_points': [1,2,3,4,5], 'labels': [1.0,2.0,3.0,4.0]}}
    kmap: dict 每个特征每一类取值组成的list及其归属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'(1, 2]': 1, '(2,3]': 2, '(3,4]': 3, '(4,5]': 4}}

    """

    def __init__(self, cate_threshold=20, specified_features='infer', bins=10, binning_method='dt', inplace=True, suffix='_bin', **kwargs):
        if suffix == '' and inplace == False:
            raise ValueError('suffix is black conflicts with inplace is False!')
            
        self.cate_threshold = cate_threshold
        self.bins = bins
        self.binning_method = binning_method
        self.inplace = inplace
        self.kwargs = kwargs
        self.suffix = suffix
        self.specified_features = specified_features

    def fit(self, X, y=None):
        if y is None:
            if self.binning_method != 'dt':
                y = [np.random.randint(0, 2) for _ in range(len(X))]
            else:
                raise ValueError('y must be not None if binning_method is dt!')

        df = pd.DataFrame(X.copy())

        target = pd.Series(y, index=df.index)

        self.columns = list(df.columns)
        if self.specified_features == 'infer':
            self.specified_columns = infer_dtypes(df, self.cate_threshold, dtype='cont')
        elif self.specified_features == 'all':
            self.specified_columns = list(df.columns)
        else:
            self.specified_columns = self.specified_features 
        
        self.map = defaultdict(dict)
        self.kmap = defaultdict(dict)
        for c in self.specified_columns:
            logger('start cont binning: %s'%c)
            self.map[c]['cut_points'] = get_cut_points(pd.to_numeric(df[c])[df[c].notnull()], target[df[c].notnull()], self.bins, self.binning_method,
                                                       **self.kwargs)
            self.map[c]['labels'] = np.arange(1, len(self.map[c]['cut_points']))
            for i in range(len(self.map[c]['cut_points']) - 1):
                self.kmap[c][i] = '(%s, %s]' % (self.map[c]['cut_points'][i], self.map[c]['cut_points'][i + 1])
        self.kmap = dict(self.kmap)
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        assert list(df.columns) == self.columns

        for c in self.specified_columns:
            df[str(c) + self.suffix] = pd.cut(pd.to_numeric(df[c]), bins=self.map[c]['cut_points'], labels=self.map[c]['labels'], precision=8).astype(float).fillna(0).astype(int).astype('category')
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

    def __init__(self, cate_threshold=200, specified_features='infer', woe_min=-4.6, woe_max=4.6, inplace=True, suffix='_woe'):
        if suffix == '' and inplace == False:
            raise ValueError('suffix is black conflicts with inplace is False!')
            
        self.cate_threshold = cate_threshold
        self.woe_min = woe_min
        self.woe_max = woe_max
        self.inplace = inplace
        self.suffix = suffix
        self.specified_features = specified_features
        
    def fit(self, X, y):
        self.map = {}
        self.ivmap = {}   
        
        df = pd.DataFrame(X.copy())
        self.columns = list(df.columns)

        if self.specified_features == 'infer':
            self.specified_columns = infer_dtypes(df, self.cate_threshold, dtype='cate')
        elif self.specified_features == 'all':
            self.specified_columns = list(df.columns)
        else:
            self.specified_columns = self.specified_features        

        target = pd.Series(y, index=df.index)
        for c in self.specified_columns:
            pos = df[c][target==1].value_counts(normalize=True, dropna=False)
            neg = df[c][target==0].value_counts(normalize=True, dropna=False)
            tmp = np.log(pos.divide(neg,fill_value=0)).replace([np.inf, -np.inf], [self.woe_max, self.woe_min])
            self.map[c] = tmp.to_dict()
            self.ivmap[c] = np.sum(pos.sub(neg, fill_value=0) * tmp)
            #不能使用np.nan in self.map[c]判断，在c为数值型时，map中有nan，但是此方法无法正确判断
             
            if self.isnumeric(tmp.sort_index().index[-1]) and np.isnan(tmp.sort_index().index[-1]):
                self.map[c] = tmp.iloc[:-1, ].to_dict()
                self.map[c][np.nan] = tmp.iloc[-1]
            else:
                self.map[c] = tmp.to_dict()
                self.map[c][np.nan] = 0               
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        assert list(df.columns) == self.columns

        for c in self.specified_columns:
            df[str(c) + self.suffix] = df[c].map(self.map[c]).astype(float).fillna(self.map[c][np.nan])
            if self.inplace & (self.suffix != ''):
                del df[c]
        return df
    
    @staticmethod
    def isnumeric(s):
        try:
            float(s)
            return True
        except:
            return False

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
    kmap: dict 每个特征每一类取值组成的list及其归属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {1: '[a, b]', 2: '[c]'}}

    """

    def __init__(self, cate_threshold=200, specified_features='infer', bins=100, binning_method='dt', inplace=True, suffix='_woebin', **kwargs):
        self.cate_threshold = cate_threshold
        self.bins = bins
        self.binning_method = binning_method
        self.inplace = inplace
        self.kwargs = kwargs
        self.suffix = suffix
        self.specified_features = specified_features
        
    def fit(self, X, y=None):
        self.kmap = {}

        if y is None:
            if self.binning_method != 'dt':
                y = [np.random.randint(0, 2) for _ in range(len(X))]
            else:
                raise ValueError('y must be not None if binning_method is dt!')

        df = pd.DataFrame(X.copy())

        self.columns = list(df.columns)
        
        if self.specified_features == 'infer':
            self.specified_columns = infer_dtypes(df, self.cate_threshold, dtype='cate')
        elif self.specified_features == 'all':
            self.specified_columns = list(df.columns)
        else:
            self.specified_columns = self.specified_features
        
        df = df[self.specified_columns]
        self.specified_woebin_columns = list(df.nunique()[df.nunique()>self.bins].index)
        self.specified_unique_columns = [c for c in self.specified_columns if c not in self.specified_woebin_columns]
        
        self.encoder = Pipeline([('WOEEncoder', WOEEncoder(specified_features='all', suffix='')), \
                                 ('ContBinningEncoder', ContBinningEncoder(cate_threshold=self.bins, specified_features='all', bins=self.bins, binning_method=self.binning_method, suffix=self.suffix, **self.kwargs))])
        logger('start cate binning')
        self.encoder.fit(df[self.specified_woebin_columns], y)

        for c in self.specified_woebin_columns:
            logger('start cate woe binning map: %s'%c)
            self.kmap[c] = self.map_dicts(self.encoder.steps[0][1].map[c], self.encoder.steps[1][1].map[c])

        for c in self.specified_unique_columns:
            logger('start cate unique binning map: %s'%c)
            l = [np.nan] + list(df[c].dropna().unique())
            self.kmap[c] = dict(zip(range(len(l)),l))
        return self

    def transform(self, X):
        if self.inplace and self.suffix == '':
            raise ValueError('inplace is True conflicts with suffix is blank!')
        df = pd.DataFrame(X.copy())
        assert list(df.columns) == self.columns

        t1 = self.encoder.transform(df[self.specified_woebin_columns])
        for c in self.specified_unique_columns:
            # TODO: 缺失和未见词应该统一为0么？还是进行区分？
            df[str(c) + self.suffix] = df[c].map({v: k for k, v in self.kmap[c].items()}).fillna(0).astype('category')
            if self.inplace:
                del df[c]
        if self.inplace:
            t2 = df[[c for c in df.columns if c not in set(self.specified_woebin_columns)]]
        else:
            t2 = df
        df = pd.concat([t2, t1], axis=1)
        return df

    @staticmethod
    def invert_dict_nonunique(d):
        newdict = defaultdict(list)
        for k, v in d.items():
            newdict[v].append(k)
        return dict(newdict)

    def map_dicts(self, d1, d2):
        """
        d1: dict, 原始特征值对应映射后值
        d2: dict, 映射后特征值分箱规则和对应箱编号
        """
        d = self.invert_dict_nonunique(d1)
        tmp = pd.cut(list(d.keys()), bins=d2['cut_points'], precision=8, labels=d2['labels'])
        d3 = self.invert_dict_nonunique(dict(zip(list(d.keys()), tmp)))

        maps = defaultdict(list)
        for k, v in d3.items():
            for val in v:
                maps[k].extend(d[val])
        return dict(maps)    

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

    def __init__(self, inplace=True, suffix='_count', specified_features='infer', cate_threshold=200):
        """
        类别变量按照出现频率编码
        :param inplace: 是否删除原始字段
        :suffix: 新字段后缀
        :specified_features: "all", "infer" or array
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
        self.specified_features = specified_features
        self.cate_threshold = 20

    def fit(self, X, y=None):
        """
        :param X: df
        :param y: None
        :return:
        """
        self.map = {}
        df = pd.DataFrame(X.copy())
        
        if self.specified_features == 'infer':
            self.specified_columns = infer_dtypes(df, self.cate_threshold, dtype='cate')
        elif self.specified_features == 'all':
            self.specified_columns = list(df.columns)
        else:
            self.specified_columns = self.specified_features    
            
        for c in self.specified_columns:
            l = [np.nan] + list(df[c].dropna().unique())
            self.map[c] = dict(zip(l, range(len(l))))

        self.columns = list(df.columns)
        return self

    def transform(self, X):
        """
        :param X: df
        :return:
        """
        df = pd.DataFrame(X.copy())
        assert list(df.columns) == self.columns

        for c in self.specified_columns:
            df[str(c) + self.suffix] = df[c].map(self.map[c]).astype(float).fillna(0)
            if self.inplace & (self.suffix != ''):
                del df[c]
        return df

class CateOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, n_values="auto", specified_features="infer",
                 dtype=np.float64, sparse=True, handle_unknown='ignore', cate_threshold=200, inplace=True, suffix='_onehot'):
        self.n_values = n_values
        self.specified_features = specified_features
        self.cate_threshold = cate_threshold
        self.dtype = dtype
        self.sparse = sparse
        self.handle_unknown = handle_unknown
        self.inplace = inplace
        self.suffix = suffix
        self.onehot_params = self.get_params()
        del self.onehot_params['specified_features']  
        del self.onehot_params['cate_threshold']
        del self.onehot_params['inplace']
        del self.onehot_params['suffix']

    def fit(self, X, y=None):
        df = pd.DataFrame(X.copy())
        self.columns = list(df.columns)
        
        if self.specified_features == 'infer':
            self.specified_columns = infer_dtypes(df, self.cate_threshold, dtype='cate')
        elif self.specified_features == 'all':
            self.specified_columns = list(df.columns)
        else:
            self.specified_columns = self.specified_features    
            
        self.encoder = Pipeline([('CateLabelEncoder', CateLabelEncoder(specified_features='all', cate_threshold=self.cate_threshold, suffix='')), \
                                 ('OneHotEncoder', OneHotEncoder(**self.onehot_params))])
        self.encoder.fit(df[self.specified_columns])
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        assert list(df.columns) == self.columns
        
        t1 = self.encoder.transform(df[self.specified_columns])
        if self.inplace:
            t2 = df[[c for c in df.columns if c not in set(self.specified_columns)]]
        else:
            t2 = df
        if not self.sparse:
            df = pd.DataFrame(t1)
            df.columns = [str(c) + self.suffix for c in df.columns]
            df = pd.concat([t2, t1], axis=1)
            return df
        else:
            return t1

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

    def __init__(self, cate_threshold=200, cate_bins=100, cont_bins=20, binning_method='dt', inplace=True, suffix='_bin', **kwargs):
        self.cate_threshold = cate_threshold
        self.cate_bins = cate_bins
        self.cont_bins = cont_bins
        self.binning_method = binning_method
        self.inplace = inplace
        self.kwargs = kwargs
        self.suffix = suffix

    def fit(self, X, y=None):
        self.kmap = {}
        df = pd.DataFrame(X.copy())

        self.columns = list(df.columns)

        logger('start column infer')
        self.cate_columns = infer_dtypes(df, self.cate_threshold, dtype='cate')
        self.cont_columns = [c for c in df.columns if c not in self.cate_columns]

        logger('start cont_columns binning')
        if len(self.cont_columns) > 0:
            self.cont_bin_enc = ContBinningEncoder(cate_threshold=self.cate_threshold, bins=self.cont_bins,
                                                   binning_method=self.binning_method, inplace=self.inplace,
                                                   suffix=self.suffix)
            self.cont_bin_enc.fit(df[self.cont_columns], y)
            self.kmap.update(self.cont_bin_enc.kmap)

        logger('start cate_columns binning')
        if len(self.cate_columns) > 0:
            self.cate_bin_enc = CateBinningEncoder(bins=self.cate_bins, cate_threshold=self.cate_threshold, inplace=self.inplace, suffix=self.suffix)
            self.cate_bin_enc.fit(df[self.cate_columns], y)
            self.kmap.update(self.cate_bin_enc.kmap)

        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())   
        assert list(df.columns) == self.columns

        df_result = pd.DataFrame()
        if len(self.cont_columns) > 0:
            df_result = pd.concat([df_result, self.cont_bin_enc.transform(df[self.cont_columns])], axis=1)
        if len(self.cate_columns) > 0:
            df_result = pd.concat([df_result, self.cate_bin_enc.transform(df[self.cate_columns])], axis=1)
        return df_result

class BinningWOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cate_threshold=200, cate_bins=100, cont_bins=10, binning_method='dt', inplace=True, woe_min=-4.6, woe_max=4.6, \
                 suffix='_binwoe', **kwargs):
        self.cate_threshold = cate_threshold
        self.cate_bins = cate_bins
        self.cont_bins = cont_bins
        self.binning_method = binning_method
        self.inplace = inplace
        self.kwargs = kwargs
        self.suffix = suffix
        self.woe_min = woe_min
        self.woe_max = woe_max

    def fit(self, X, y=None):
        steps = [
            ('BinningEncoder', BinningEncoder(cate_threshold=self.cate_threshold, cate_bins=self.cate_bins, cont_bins=self.cont_bins, binning_method=self.binning_method,
                                       suffix='', inplace=self.inplace, **self.kwargs)),
            ('WOEEncoder', WOEEncoder(cate_threshold=self.cate_threshold, woe_min=self.woe_min, woe_max=self.woe_max,
                                suffix=self.suffix, inplace=self.inplace))]
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X, y)
        return self

    def transform(self, X):
        return self.pipeline.transform(X)