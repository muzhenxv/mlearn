"""
Use this class to process categorical variables.
document: https://www.slideshare.net/HJvanVeen/feature-engineering-72376750
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import bisect
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.preprocessing import Imputer
from collections import defaultdict
from sklearn.base import BaseEstimator, TransformerMixin


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


class ContBinningEncoder(BaseEstimator, TransformerMixin):
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

    Attributes
    ----------
    map: dict 每个特征每个取值所属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'cut_points': [1,2,3,4,5], 'labels': [1.0,2.0,3.0,4.0]}}
    kmap: dict 每个特征每一类取值组成的list及其归属的箱，如果没有参与分箱则该特征不在dict中 like {'feature1': {'(1, 2]': 1, '(2,3]': 2, '(3,4]': 3, '(4,5]': 4}}

    """

    def __init__(self, diff_thr=20, bins=10, binning_method='dt', inplace=True, suffix='_bin', **kwargs):
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
        if (y is None) & (self.binning_method == 'dt'):
            self.binning_method = 'qcut'

        target = pd.DataFrame(y)
        target.columns = ['label']

        self.columns = list(df.columns)

        t = df.nunique()
        self.bin_cols = list(t[t > self.diff_thr].index)
        self.notbin_cols = list(t[t <= self.diff_thr].index)

        self.map = defaultdict(dict)
        self.kmap = defaultdict(dict)
        for c in self.bin_cols:
            tmp = pd.concat([df[c], target], axis=1)
            tmp = tmp.dropna()

            self.map[c]['cut_points'] = get_cut_points(tmp[c], tmp['label'], self.bins, self.binning_method,
                                                       **self.kwargs)
            # labels需要转成float，因为最终可能出现一些case对应label是np.nan, 如果是int，可能在map时由于类型问题配对不上
            self.map[c]['labels'] = np.arange(len(self.map[c]['cut_points']) - 1) * 1.0
            # TODO: why 上一句转成float， 这里for loop 用int？
            for i in range(len(self.map[c]['cut_points']) - 1):
                self.kmap[c]['(%s, %s]' % (self.map[c]['cut_points'][i], self.map[c]['cut_points'][i + 1])] = i
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        for c in self.bin_cols:
            # 可能有nan，所以转换为float
            df[c + self.suffix] = pd.cut(df[c], bins=self.map[c]['cut_points'], labels=self.map[c]['labels']).astype(
                float)
            if self.inplace & (self.suffix != ''):
                del df[c]

        return df


class ContImputerEncoder(Imputer):
    """
    此类继承自sklearn.preprocessing.Imputer，fit与基类完全一致，transform方法返回变为pandas.DataFrame。

    仅适用于cont， 支持缺失值

    Parameters
    ----------
    missing_values : integer or "NaN", optional (default="NaN")
        The placeholder for the missing values. All occurrences of
        `missing_values` will be imputed. For missing values encoded as np.nan,
        use the string value "NaN".

    strategy : string, optional (default="mean")
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          the axis.
        - If "median", then replace missing values using the median along
          the axis.
        - If "most_frequent", then replace missing using the most frequent
          value along the axis.

    axis : integer, optional (default=0)
        The axis along which to impute.

        - If `axis=0`, then impute along columns.
        - If `axis=1`, then impute along rows.

    verbose : integer, optional (default=0)
        Controls the verbosity of the imputer.

    copy : boolean, optional (default=True)
        If True, a copy of X will be created. If False, imputation will
        be done in-place whenever possible. Note that, in the following cases,
        a new copy will always be made, even if `copy=False`:

        - If X is not an array of floating values;
        - If X is sparse and `missing_values=0`;
        - If `axis=0` and X is encoded as a CSR matrix;
        - If `axis=1` and X is encoded as a CSC matrix.

    Attributes
    ----------
    enc : 实例化的Imputer对象

    enc.statistics_ : array of shape (n_features,)
        The imputation fill value for each feature if axis == 0.

    Notes
    -----
    - When ``axis=0``, columns which only contained missing values at `fit`
      are discarded upon `transform`.
    - When ``axis=1``, an exception is raised if there are rows for which it is
      not possible to fill in the missing values (e.g., because they only
      contain missing values).
    """

    def fit(self, X, y=None):
        df = pd.DataFrame(X.copy())
        self.columns = list(df.columns)
        self.enc = Imputer(**self.get_params())
        self.enc.fit(df)
        return self

    def transform(self, X):
        df = pd.DataFrame(X.copy())
        if list(df.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        df = pd.DataFrame(self.enc.transform(df))
        df.columns = self.columns
        return df
