import pandas as pd
from sklearn.feature_selection import *
from ..data_service import get_data
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.naive_bayes import *
from sklearn.tree import *
from sklearn.svm import *
from sklearn.naive_bayes import *
from sklearn.decomposition import *
from sklearn.random_projection import *
from sklearn.covariance import *
from sklearn.manifold import *
from sklearn.gaussian_process import *
from sklearn.calibration import *
from sklearn.discriminant_analysis import *
from xgboost.sklearn import XGBClassifier
from ..monitor.data_monitor import *
from ..base_utils import instantiate_utils as instantiate
from ..base_utils.base_utils import get_feature_importances

from sklearn.feature_selection.rfe import if_delegate_has_method
from sklearn.feature_selection.rfe import clone
from sklearn.utils import safe_sqr
import statsmodels.api as sm


class SklearnFilter(BaseEstimator, TransformerMixin):
    def __init__(self, method, params):
        self.method = method
        self.params = params
        if 'score_func' in self.params:
            self.indicator = self.params['score_func']
            self.params['score_func'] = eval(self.params['score_func'])
        elif 'estimator' in self.params:
            self.indicator = self.params['estimator']['method']
            self.params['estimator'] = instantiate.instantiate(self.params['estimator']['method'],
                                                               self.params['estimator']['params'])

        if 'n_features_to_select' in self.params:
            self.n_features_to_select = self.params['n_features_to_select']
            if self.method != 'RFE':
                self.params.pop('n_features_to_select')
        else:
            self.n_features_to_select = None

    def fit(self, X_train, y_train=None, X_test=None, y_test=None):
        """
        加入X_test,y_test是为了reweight时需要用到test集考虑
        :param X_train:
        :param y_train:
        :param X_test:
        :param y_test:
        :return:
        """
        df = X_train.copy()
        df.fillna(-999, inplace=True)
        label = y_train.copy()

        self.columns = list(df.columns)

        # self.enc = eval(self.method)(**self.params)
        self.enc = instantiate.instantiate(self.method, self.params)
        self.enc.fit(df, label)

        self.features_report = self._get_report(self.enc, self.columns, self.method, self.indicator)

        if self.n_features_to_select:
            self.selected_columns = list(self.features_report.index)[:self.n_features_to_select]
        else:
            self.selected_columns = list(self.features_report.index[self.features_report[self.indicator + '_support']])
        return self

    def transform(self, X_train):
        if list(X_train.columns) != self.columns:
            c = [c for c in self.columns if c not in df.columns]
            raise ValueError(f'Unexpected columns {c} are found!')

        df_final = X_train[self.selected_columns]
        return df_final

    def fit_transform(self, X_train, y_train=None, X_test=None, y_test=None):
        return self.fit(X_train, y_train, X_test, y_test).transform(X_train)

    @staticmethod
    def _get_report(enc, columns, method, indicator=''):
        df = pd.DataFrame(columns, columns=['feature_name'])
        if method == 'SelectKBest':
            df[indicator + '_support'] = enc.get_support()
            df[indicator + '_pvalues'] = enc.pvalues_
            df[indicator + '_scores'] = enc.scores_
            df = df.sort_values(indicator + '_pvalues', ascending=True)
        elif method in ['RFE', 'BaseRFEFilter', 'LRFilter', 'BaseLRFilter']:
            df[indicator + '_support'] = enc.get_support()
            df[indicator + '_ranking'] = enc.ranking_
            df[indicator + '_feature_importances'] = np.nan
            df[indicator + '_feature_importances'][enc.get_support()] = get_feature_importances(enc.estimator_)
            df = df.sort_values(indicator + '_feature_importances', ascending=False)
        elif method == 'SelectFromModel':
            df[indicator + '_support'] = enc.get_support()
            df[indicator + '_feature_importances'] = get_feature_importances(enc.estimator_)
            df = df.sort_values(indicator + '_feature_importances', ascending=False)
        else:
            raise ValueError(f'Unexpected method {method} are found!')

        df = df.set_index('feature_name')
        return df


class BaseLRFilter(RFE):
    """Feature ranking with recursive feature elimination.
    """

    def __init__(self, estimator, n_features_to_select=20, step=1, verbose=0, alpha=0.05, method='coef', **kwargs):
        """

        :param estimator:
        :param n_features_to_select:
        :param step:
        :param verbose:
        :param alpha:
        :param method: 'coef': 特征woe，按系数筛选，系数为负删除（woe之后系数为负说明单特征区分方向和在模型中方向相反，较为反常）， 'stepwise'： 正向逐布回归，不停加特征直到无增益， 'RFE'：skleran.RFE
        :param stepwise:
        :param kwargs:
        """
        super().__init__(estimator, step=step, verbose=verbose,
                         n_features_to_select=n_features_to_select)
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.alpha = alpha
        self.verbose = verbose

    def fit(self, X, y):
        self.meta_columns = X.columns.tolist()
        self.support_ = np.array([True for col in self.meta_columns])

        if self.method == 'coef':
            return self._fit_by_coef(X, y)
        elif self.method == 'stepwise':
            return self._fit_by_stepwise(X, y, n_features_to_select=self.n_features_to_select, alpha=self.alpha)
        else:
            return self._fit(X, y)

    def _fit_by_coef(self, X, y):
        # TODO : fit多次，取coef的平均值来度量是否要删除该特征

        X_copy = X.copy()
        n_features = len(self.meta_columns)
        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        estimator = clone(self.estimator)
        estimator.fit(X_copy, y)
        coef_map = dict(zip(X_copy.columns.tolist(), list(estimator.coef_[0])))

        # Elimination by estimator coeffication
        while np.sum(estimator.coef_ < 0) > 0:
            # 是否需要考虑特殊情况
            features_selected = [k for k, v in coef_map.items() if v > 0]
            X_copy = X_copy[features_selected]
            estimator.fit(X_copy, y)

            # Get coefs
            coef_map = dict(zip(X_copy.columns.tolist(), list(estimator.coef_[0])))
            coefs = estimator.coef_

            # Get ranks
            if coefs.ndim > 1:
                ranks = np.argsort(safe_sqr(coefs).sum(axis=0))
            else:
                ranks = np.argsort(safe_sqr(coefs))
            ranks = np.ravel(ranks)

            support_ = np.array([True if col in coef_map.keys() else False for col in self.meta_columns])
            ranking_[np.logical_not(support_)] += 1

        # Set final attributes
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._transform(X), y)
        return self

    def _fit_by_stepwise(self, X, y, n_features_to_select=20, alpha=0.05, lr_type='logistic'):
        # TODO :
        if n_features_to_select > len(self.meta_columns):
            n_features_to_select = len(self.meta_columns)

        n_features = len(self.meta_columns)
        support_ = np.ones(n_features, dtype=np.bool)
        ranking_ = np.ones(n_features, dtype=np.int)

        remaining = set(X.columns)
        selected = []
        exclude_cols = []
        current_score = 0
        best_new_score = 0

        # Forford select
        while remaining and best_new_score <= alpha and len(selected) < n_features_to_select:
            scores_with_candidates = []
            for candidate in remaining:
                if candidate in exclude_cols:
                    continue
                x_candidates = selected + [candidate]
                try:
                    if lr_type == 'logistic':
                        model_stepwise_forward = sm.Logit(y, X[x_candidates]).fit(disp=False)
                    elif lr_type == 'linear':
                        model_stepwise_forward = sm.OLS(y, X[x_candidates]).fit(disp=False)
                    else:
                        raise ValueError(f'Unexpected model type {lr_type} are found!')
                except:
                    exclude_cols.append(candidate)
                    x_candidates.remove(candidate)
                    continue
                score = model_stepwise_forward.pvalues[candidate]
                scores_with_candidates.append((score, candidate))

            scores_with_candidates.sort(reverse=True)
            best_new_score, best_candidate = scores_with_candidates.pop()
            if best_new_score <= alpha:
                remaining.remove(best_candidate)
                selected.append(best_candidate)
                if self.verbose:
                    print(best_candidate + ' enters: pvalue: ' + str(best_new_score))

            # Get supports and ranking
            support_ = np.array([True if col in selected else False for col in self.meta_columns])
            ranking_[np.logical_not(support_)] += 1

        if lr_type == 'logistic':
            model_stepwise_backford = sm.Logit(y, X[selected]).fit(disp=False)
        elif lr_type == 'linear':
            model_stepwise_backford = sm.OLS(y, X[selected]).fit(disp=False)
        else:
            raise ValueError(f'Unexpected model type {lr_type} are found!')

        # Backford select
        for fea in selected:
            if model_stepwise_backford.pvalues[fea] > alpha:
                selected.remove(fea)
                if self.verbose:
                    print(fea + ' removed: pvalue: ' + str(model_stepwise_backford.pvalues[fea]))

        if sum(support_) != len(selected):
            support_ = np.array([True if col in selected else False for col in self.meta_columns])
            ranking_[np.logical_not(support_)] += 1

        # Set final attributesstepwise
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(self._transform(X), y)
        return self

    def _transform(self, X):
        # TODO: check support
        return X.loc[:, self.support_]

    def get_columns(self):
        tmp = list(zip(self.meta_columns, self.support_))
        cols = [c[0] for c in tmp if c[1]]
        return cols


class SelectKBestFilter(SklearnFilter):
    def __init__(self, **kwargs):
        super().__init__('SelectKBest', kwargs)


class SelectFromModelFilter(SklearnFilter):
    def __init__(self, **kwargs):
        super().__init__('SelectFromModel', kwargs)


class RFEFilter(SklearnFilter):
    def __init__(self, **kwargs):
        super().__init__('RFE', kwargs)


class LRFilter(SklearnFilter):
    def __init__(self, **kwargs):
        super().__init__('BaseLRFilter', kwargs)


class StableFilter(BaseEstimator, TransformerMixin):
    """
    特征稳定性判断，基于稳定性筛选
    """

    def __init__(self, indice_name='psi', indice_thr=0.2):
        self.indice_name = indice_name
        self.indice_thr = indice_thr

    def fit(self, X_train, y_train=None, X_test=None, y_test=None):
        dic, mcc, weights = consistent_dataset_test(X_train, X_test)
        df = pd.DataFrame(dic).T
        indice_col = df.columns[df.columns.astype(str).str.startswith(self.indice_name)][0]
        self.exclude_cols = list(df.index[df[indice_col] > self.indice_thr])
        df[indice_col + '_support'] = df[indice_col] <= self.indice_thr
        self.exclude_cols = list(df.index[~df[indice_col + '_support']])
        self.features_report = df
        return self

    def transform(self, X_train):
        return X_train.drop(self.exclude_cols, axis=1)

    def fit_transform(self, X_train, y_train=None, X_test=None, y_test=None):
        return self.fit(X_train, y_train, X_test, y_test).transform(X_train)
