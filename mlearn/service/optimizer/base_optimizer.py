from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score
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
from sklearn.model_selection import train_test_split
from sklearn.metrics.scorer import check_scoring
import pandas as pd
from ..base_utils import *

def BayesianOptimizer(df, target, estimator, n_folds=3, test_size=0.2, score_func='roc_auc', gp_params=None):
    """
    贝叶斯优化

    Parameters
    ----------
    score_func : str, default: 'roc_auc'
         Valid options are ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision',
                            'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
                            'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score',
                            'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error',
                            'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score',
                            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
                            'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples',
                            'recall_weighted', 'roc_auc', 'v_measure_score']

    gp_params : dict, default: None

    """
    method = estimator['method']
    params = estimator['params']
    if n_folds > 1:
        def clf_opt(**params):
            for k, v in params.items():
                if v > 1:
                    params[k] = int(v)
            val = cross_val_score(instantiate(method, params), df, target, scoring=score_func, cv=n_folds).mean()
            return val
    else:
        tr, te, tr_y, te_y = train_test_split(df, target, test_size=test_size, random_state=7)

        def clf_opt(**params):
            for k, v in params.items():
                if v >= 1:
                    params[k] = int(v)
            clf = instantiate(method, params)
            clf.fit(tr, tr_y)
            val = check_scoring(clf, score_func)(clf, te, te_y)
            return val

    BO = BayesianOptimization(clf_opt, params)

    BO.maximize(**gp_params)

    t = pd.DataFrame(BO.res['all'])
    t.columns = [i if i == 'params' else score_func for i in t.columns]
    all_df = pd.concat([t.params.apply(pd.Series), t[score_func]], axis=1)
    all_df = all_df.applymap(lambda s: s if s < 1 else int(s))

    max_dict = BO.res['max']

    return all_df, max_dict, BO


def GSOptimizer(df, target):
    return None
