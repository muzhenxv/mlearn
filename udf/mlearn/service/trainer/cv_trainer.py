import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import ks_2samp
from sklearn import metrics as mr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import *
# from sklearn.ensemble import *
# from sklearn.naive_bayes import *
# from sklearn.tree import *
# from sklearn.svm import *
# from sklearn.naive_bayes import *
# from sklearn.decomposition import *
# from sklearn.random_projection import *
# from sklearn.covariance import *
# from sklearn.manifold import *
# from sklearn.gaussian_process import *
# from sklearn.calibration import *
# from sklearn.discriminant_analysis import *
# from xgboost.sklearn import XGBClassifier
# from ...algorithms import SkopeRuler
import pickle
import json
from ..monitor import data_monitor as dm
import random
from ..base_utils import instantiate_utils as iu

default_params = {
    "colsample_bytree": 0.8,
    "reg_lambda": 20,
    "silent": True,
    "base_score": 0.5,
    "scale_pos_weight": 1,
    "eval_metric": "auc",
    "max_depth": 3,
    "n_jobs": 1,
    "early_stopping_rounds": 30,
    "n_estimators": 1000,
    "random_state": 0,
    "reg_alpha": 1,
    "booster": "gbtree",
    "objective": "binary:logistic",
    "verbose": False,
    "colsample_bylevel": 0.8,
    "subsample": 0.7,
    "learning_rate": 0.1,
    "gamma": 0.5,
    "max_delta_step": 0,
    "min_child_weight": 10
}


def base_model_evaluation(X_train, y_train, X_test=None, y_test=None, method='XGBClassifier', params=default_params,
                          n_folds=5, test_size=0, random_state=7, random_seed=7, shift_thr=0.2,
                          oversample=False, verbose=True, reweight=False, reweight_with_label=False,
                          cut_off_use_weights=True, sample_weights=None, cut_off_sample_ratio=None, report_dst=None):
    """
    测试集入参是为了reweight
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param method:
    :param params:
    :param n_folds:
    :param test_size:
    :param random_state:
    :param shift_thr:
    :param oversample:
    :param verbose:
    :param reweight:
    :param reweight_with_label:
    :param sample_weights: array or list or Series
    :param cut_off_sample_ratio:
    :param report_dst:
    :return:
    """
    random.seed(random_seed)
    df = X_train.copy()
    target = y_train.copy()

    if sample_weights is not None:
        if X_train.shape[0] != len(sample_weights):
            raise ValueError('the length of sample_weights if different from sample_weights!')
        mcc = None
    else:
        if X_test is None:
            reweight = False
        if y_test is None:
            reweight_with_label = False

        if reweight:
            if reweight_with_label:
                df1 = pd.concat([X_train, y_train], axis=1)
                df2 = pd.concat([X_test, y_test], axis=1)
            else:
                df1 = X_train
                df2 = X_test
            mcc, sample_weights = dm.detect_covariate_shift(df1, df2, report_dst)
            print('Covariate shift: ', mcc, 'shift_thr: ', shift_thr)
            print('=============================================================', sample_weights.shape)
            if mcc < shift_thr:
                sample_weights = None
        else:
            sample_weights = None
            mcc = None
    if sample_weights is not None:
        sample_weights_copy = sample_weights.copy()
    else:
        sample_weights_copy = None

    if (sample_weights is not None) & (cut_off_sample_ratio is not None):
        if cut_off_sample_ratio <= 1:
            cut_off_sample_ratio = int(X_train.shape[0] * cut_off_sample_ratio)
        tmp = pd.Series(sample_weights)
        tmp.index = df.index
        indice = list(tmp.sort_values(ascending=False)[:cut_off_sample_ratio].index)
        random.shuffle(indice)
        print('==================================================原始训练样本数：', df.shape[0])
        df = df.loc[indice]
        print('==================================================截断训练样本数：', df.shape[0])
        target = target.loc[indice]
        sample_weights_copy = tmp.copy()
        if cut_off_use_weights:
            sample_weights = tmp.loc[indice]
            sample_weights = sample_weights / np.mean(sample_weights)
        else:
            sample_weights = None

    if (test_size == 0) & (n_folds is None):
        raise Exception("Error: test_size and n_folds can't both invalid.")

    col_name = 'y_true'
    best_iteration = 0

    # TODO: oversample目前只对xgb有效
    if oversample:
        pn_ratio = np.sum(target == 0) / np.sum(target == 1)
        if 'scale_pos_weight' in params:
            params['scale_pos_weight'] = pn_ratio

    if test_size > 0:
        if sample_weights is not None:
            train, test, train_y, test_y, train_sample_weights, test_sample_weights = train_test_split(df, target,
                                                                                                       sample_weights,
                                                                                                       test_size=test_size,
                                                                                                       random_state=random_state)
        else:
            train, test, train_y, test_y = train_test_split(df, target,
                                                            test_size=test_size,
                                                            random_state=random_state)
            train_sample_weights = None
    else:
        train = df
        train_y = target
        train_sample_weights = sample_weights

    # random.seed(random_seed)
    # indice = list(train.index)
    # random.shuffle(indice)
    # train = train.loc[indice]
    # train_y = train_y.loc[indice]

    dic_cv = []

    if df.shape[1] == 1:
        if 'colsample_bytree' in params:
            params['colsample_bytree'] = 1

    if method == 'XGBClassifier':
        eval_metric = params.pop('eval_metric', 'auc')
        cv_verbose_eval = params.pop('verbose', False)
        early_stopping_rounds = params.pop('early_stopping_rounds', 30)

    if n_folds:
        df_val = pd.DataFrame(index=train.index)
        df_val['y_true'] = 0
        df_val['y_pred'] = 0
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        for t_index, v_index in skf.split(train, train_y):
            tra, val = train.iloc[t_index, :], train.iloc[v_index, :]
            tra_y, val_y = train_y.iloc[t_index], train_y.iloc[v_index]
            if train_sample_weights is not None:
                try:
                    tra_w = train_sample_weights.iloc[t_index]
                except:
                    tra_w = train_sample_weights[t_index]
            else:
                tra_w = None

            if method == 'XGBClassifier':
                # clf = eval(method)(**params)
                clf = iu.instantiate(method, params)
                clf.fit(tra, tra_y, eval_set=[(tra, tra_y), (val, val_y)], eval_metric=eval_metric,
                        early_stopping_rounds=early_stopping_rounds, verbose=cv_verbose_eval, sample_weight=tra_w)
                try:
                    print('best_iteration: ', clf.best_iteration)
                    best_iteration += clf.best_iteration
                except:
                    best_iteration += clf.n_estimators
            else:
                # clf = eval(method)(**params)
                clf = iu.instantiate(method, params)
                clf.fit(tra, tra_y, sample_weight=tra_w)

            if test_size > 0:
                temp = clf.predict_proba(test)[:, 1]
                dic_res = {'train_auc': roc_auc_score(tra_y, clf.predict_proba(tra)[:, 1]),
                           'val_auc': roc_auc_score(val_y, clf.predict_proba(val)[:, 1]),
                           'test_auc': roc_auc_score(test_y, temp)}
            else:
                dic_res = {'train_auc': roc_auc_score(tra_y, clf.predict_proba(tra)[:, 1]),
                           'val_auc': roc_auc_score(val_y, clf.predict_proba(val)[:, 1])}

            # val_df = pd.DataFrame({'y_true': val_y, 'y_pred': clf.predict_proba(val)[:, 1]})
            # df_val = pd.concat([df_val, val_df], axis=0)

            df_val.iloc[v_index, 0] = val_y
            df_val.iloc[v_index, 1] = clf.predict_proba(val)[:, 1]

            print(dic_res)
            dic_cv.append(dic_res)

        df_cv = cmpt_cv(dic_cv)
    else:
        df_cv = pd.DataFrame()
        df_val = pd.DataFrame()

    if test_size == 0:
        dvalid = (train, train_y)
        best_iteration = best_iteration // n_folds + 1
        early_stopping_rounds = None
    else:
        dvalid = (test, test_y)
        try:
            best_iteration = best_iteration // n_folds + 1
            early_stopping_rounds = None
        except:
            best_iteration = None

    if method == 'XGBClassifier':
        watchlist = [(train, train_y), dvalid]
        if best_iteration is not None:
            params['n_estimators'] = best_iteration
        best_iteration = params['n_estimators']
        # clf = eval(method)(**params)
        clf = iu.instantiate(method, params)
        clf.fit(train, train_y, eval_set=watchlist, eval_metric=eval_metric,
                early_stopping_rounds=early_stopping_rounds, verbose=verbose, sample_weight=train_sample_weights)
        print('best_iteration', best_iteration)
        print('early_stopping_rounds', early_stopping_rounds)
    else:
        # clf = eval(method)(**params)
        clf = iu.instantiate(method, params)
        clf.fit(train, train_y, sample_weight=train_sample_weights)

    if test_size > 0:
        pred_test = clf.predict_proba(test)[:, 1]
        df_test = pd.DataFrame({col_name: test_y, 'y_pred': pred_test})
    else:
        df_test = df_val
    pred_train = clf.predict_proba(train)[:, 1]
    df_train = pd.DataFrame({col_name: train_y, 'y_pred': pred_train})

    if df_val.shape[0] == 0:
        df_val = df_test
    return clf, df_cv, df_test, df_val, df_train, mcc, sample_weights_copy


def cmpt_cv(dic_cv):
    df_cv = pd.DataFrame(dic_cv)
    df_cv = df_cv.describe().loc[['mean', 'std', 'min', 'max']]
    return df_cv
