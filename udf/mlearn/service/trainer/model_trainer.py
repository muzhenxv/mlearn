from .cv_trainer import base_model_evaluation
from ..data_service import get_data
import xgboost as xgb
import pandas as pd


class ModelTrainer:
    """
    trianer
    """

    def __init__(self, estimator, n_folds=5, test_size=0.2, random_state=7, random_seed=7, shift_thr=0.2,
                 verbose=True, oversample=False, reweight=False, reweight_with_label=False, cut_off_use_weights=False,
                 sample_weights=None, cut_off_sample_ratio=None, report_dst=None):
        self.estimator_params = estimator.copy()
        self.method = estimator['method']
        self.params = estimator['params']
        self.n_folds = n_folds
        self.test_size = test_size
        self.random_state = random_state
        self.verbose = verbose
        self.oversample = oversample
        self.shift_thr = shift_thr
        self.reweight = reweight
        self.reweight_with_label = reweight_with_label
        self.random_seed = random_seed
        self.cut_off_use_weights = cut_off_use_weights
        self.sample_weights = sample_weights
        self.cut_off_sample_ratio = cut_off_sample_ratio
        self.report_dst = report_dst

    def fit(self, df_src, label, test_src=None, test_label=None, ret=True):
        self.label = label
        df = get_data(df_src)
        target = df.pop(self.label)

        if test_src is not None:
            test = get_data(test_src)
            test_y = test.pop(test_label)
        else:
            test = None
            test_y = None

        # TODO: why copy?dict在函数内部会被修改
        params = self.params.copy()

        self.enc, df_cv, df_test, df_val, df_train, mcc, sample_weights = base_model_evaluation(df, target, test,
                                                                                                test_y, self.method,
                                                                                                params,
                                                                                                self.n_folds,
                                                                                                self.test_size,
                                                                                                self.random_state,
                                                                                                self.random_seed,
                                                                                                self.shift_thr,
                                                                                                self.oversample,
                                                                                                self.verbose,
                                                                                                self.reweight,
                                                                                                self.reweight_with_label,
                                                                                                self.cut_off_use_weights,
                                                                                                self.sample_weights,
                                                                                                self.cut_off_sample_ratio,
                                                                                                self.report_dst)
        self.columns = list(df.columns)
        if ret:
            return self.enc, df_cv, df_test, df_val, df_train, mcc, sample_weights
        else:
            return self

    def transform(self, df_src):
        df = get_data(df_src)

        try:
            df_label = df.pop(self.label)
            df_label.name = 'y_true'
        except:
            df_label = pd.DataFrame()

        df = df[self.columns]

        pred = self.enc.predict_proba(df)[:, 1]

        df_final = pd.DataFrame(pred, columns=['y_pred'])
        df_final.index = df.index
        df_final = pd.concat([df_final, df_label], axis=1)
        return df_final

    def fit_transform(self, df_src, label, test_src=None, test_label=None):
        return self.fit(df_src, label, test_src, test_label, ret=False).transform(df_src)
