import scipy
from scipy.stats import ks_2samp, ttest_ind
from .psi import Psi
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import traceback
from ..reporter import data_reporter as dr
import pickle
from ..trainer import ModelTrainer
import os


def consistent_test(x, y, **kwargs):
    na_x_ratio = round(float(pd.Series(x).isnull().mean()), 4)
    na_y_ratio = round(float(pd.Series(y).isnull().mean()), 4)
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    l1 = len(x) - 1
    l2 = len(y) - 1
    try:
        ftest_pvalue = round(float(scipy.stats.f.sf(np.var(x) / np.var(y), l1, l2)), 4)
    except:
        ftest_pvalue = np.nan
        # try:
        #     ftest_pvalue = 'Error Var(%s, %s)' % (np.var(x), np.var(y))
        # except Exception as e:
        #     ftest_pvalue = repr(e)

    try:
        ttest_pvalue = round(float(ttest_ind(x, y)[1]), 4)
    except Exception as e:
        ttest_pvalue = np.nan

    try:
        kstest_pvalue = round(float(ks_2samp(x, y)[1]), 4)
    except Exception as e:
        kstest_pvalue = np.nan

    try:
        psi = Psi(**kwargs)
        psi.fit(x)
        psi_value, extreme_ratio = psi.transform(y)
        psi_value = round(float(psi_value), 4)
        extreme_ratio = round(float(extreme_ratio), 4)
    except Exception as e:
        psi_value, extreme_ratio = np.nan, np.nan

    dic = {'ftest-pvalue': ftest_pvalue, 'ttest-pvalue': ttest_pvalue,
           'kstest-pvalue': kstest_pvalue, 'psi': psi_value,
           'extreme_ratio': extreme_ratio, 'null ratio(train, test)': '(%s, %s)' % (na_x_ratio, na_y_ratio)}
    return dic


def detect_covariate_shift(train, test, report_dst=None):
    df = pd.concat([pd.DataFrame(train), pd.DataFrame(test)], axis=0)
    y_col = 'label'
    while y_col in df.columns:
        y_col += 'l'
    df[y_col] = np.concatenate([np.ones(len(train)), np.zeros(len(test))])

    params = {
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
    estimator_params = {'method': 'XGBClassifier', 'params': params}

    enc = ModelTrainer(estimator=estimator_params, n_folds=5, test_size=0)
    clf, df_cv, df_test, df_val, df_train, mcc, sample_weights = enc.fit(df, y_col)

    predictions = df_val['y_pred'].iloc[:train.shape[0]]
    weights = (1. / predictions) - 1.
    weights /= np.mean(weights)

    df_val_cp = df_val.copy()
    df_val_cp['y_pred'] = (df_val_cp['y_pred'] > df_val_cp['y_true'].mean()).astype(int)
    mcc = matthews_corrcoef(df_val_cp['y_true'], df_val_cp['y_pred'])

    if report_dst is not None:
        name = 'trainer'
        train_data_dst = os.path.join(report_dst, name, 'train')
        train_report_dst = os.path.join(report_dst, name, 'report')
        if not os.path.exists(train_data_dst):
            os.makedirs(train_data_dst)
        if not os.path.exists(train_report_dst):
            os.makedirs(train_report_dst)

        train_train_dst = os.path.join(train_data_dst, f'{name}_train_result.pkl')
        train_val_dst = os.path.join(train_data_dst, f'{name}_val_result.pkl')
        train_test_dst = os.path.join(train_data_dst, f'{name}_test_result.pkl')
        train_cv_dst = os.path.join(train_data_dst, f'{name}_cv_result.pkl')
        train_process_dst = os.path.join(train_data_dst, f'{name}_enc.pkl')
        mcc_dst = os.path.join(train_data_dst, f'{name}_mcc.pkl')
        sample_weights_dst = os.path.join(train_data_dst, f'{name}_sample_weights.pkl')

        pickle.dump(enc, open(train_process_dst, 'wb'))
        df_cv.to_pickle(train_cv_dst)
        df_val.to_pickle(train_val_dst)
        df_train.to_pickle(train_train_dst)
        df_test.to_pickle(train_test_dst)
        pickle.dump(mcc, open(mcc_dst, 'wb'))
        pickle.dump(weights, open(sample_weights_dst, 'wb'))

        dr.trainer_report(enc, df_train, df_val, df_test, 'test', report_dst)
    return mcc, weights


def consistent_dataset_test(X1, X2, label=None, report_dst=None, exclude_cols=None, covariate_shift_eva=False,
                            label_f=False):
    df1 = X1.copy()
    df2 = X2.copy()
    set_cols1 = (set(df1.columns) - set(df2.columns))
    set_cols2 = (set(df2.columns) - set(df1.columns))
    if set_cols1 | set_cols2:
        raise ValueError('%s exist in only one dataframe!' % (set_cols1 | set_cols2))

    dic = {}
    if exclude_cols is not None:
        cols = [c for c in df1.columns if c not in exclude_cols]
    else:
        cols = list(df1.columns)
    for c in cols:
        if c != label:
            try:
                c_v = consistent_test(df1[c], df2[c])
            except:
                c_v = np.nan
            dic[c] = c_v

    if not label_f:
        if label in df1.columns:
            del df1[label]
        if label in df2.columns:
            del df2[label]

    if covariate_shift_eva:
        try:
            mcc, weights = detect_covariate_shift(df1, df2, report_dst)
        except Exception as e:
            print(traceback.format_exc())
            mcc, weights = np.nan, np.nan
    else:
        mcc, weights = np.nan, np.nan
    return dic, mcc, weights


if __name__ == '__main__':
    x = np.random.normal(1, 1, 1000)
    y = np.random.normal(1, 1, 1000)
    consistent_test(x, y)

    test_path = '/Users/muzhen/repo/mlearn/mlearn/data/chaintest_data.csv'
    df = pd.read_csv(test_path, index_col=0)
    df1 = df.iloc[:5000, :20]
    df2 = df.iloc[5000:, :20]
    dic, mcc, weights = consistent_dataset_test(df1, df2, exclude_cols=['apply_risk_id', 'apply_risk_created_at'])
