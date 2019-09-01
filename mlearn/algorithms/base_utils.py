import numpy as np
import pandas as pd

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from sklearn import metrics
import copy


class RulerMixin:
    _estimator_type = 'ruler'

    def _single_rule_vote(self, X, rule):
#         df = check_array(X)
        df = pd.DataFrame(X, columns=self.feature_names_)

        scores = np.zeros(X.shape[0], dtype=int)
        scores[list(df.query(rule, engine='python').index)] += 1
        return scores

    def _get_metrics(self, y_true, y_pred, average='binary', describe=''):

        l_1 = self.get_lift(y_true, y_pred, pos_label=1, average=average)
        l_0 = self.get_lift(y_true, y_pred, pos_label=0, average=average)
        p_1 = metrics.precision_score(y_true, y_pred, pos_label=1, average=average)
        p_0 = metrics.precision_score(y_true, y_pred, pos_label=0, average=average)
        r_1 = metrics.recall_score(y_true, y_pred, pos_label=1, average=average)
        r_0 = metrics.recall_score(y_true, y_pred, pos_label=0, average=average)
        f1_1 = metrics.f1_score(y_true, y_pred, pos_label=1, average=average)
        f1_0 = metrics.f1_score(y_true, y_pred, pos_label=0, average=average)
        c_1 = y_true.sum()
        h_1 = (y_true & y_pred).sum()
        c_0 = (1 - np.array(y_true)).sum()
        h_0 = ((1 - np.array(y_true)) & (1 - np.array(y_pred))).sum()
        return [l_1, p_1, r_1, f1_1, h_1, c_1, l_0, p_0, r_0, f1_0, h_0, c_0, describe]

    @staticmethod
    def get_lift(y_true, y_pred, pos_label=1, average='binary'):
        """
        TODO: 考虑分母为0的情况
        """
        y_true_cp = copy.deepcopy(y_true)
        y_pred_cp = copy.deepcopy(y_pred)
        if pos_label == 0:
            y_true_cp = 1 - np.array(y_true_cp)
            y_pred_cp = 1 - np.array(y_pred_cp)
        try:
            lift = ((y_true_cp & y_pred_cp).sum() / y_pred_cp.sum()) / y_true_cp.mean()
        except:
            lift = np.nan
        return lift

    def get_metrics(self, X, y, average='binary'):
        if (type(X) == pd.DataFrame) & (list(X.columns) != self.feature_names_):
            raise ('columns inconsistence!')

#         df = check_array(X)
        df = pd.DataFrame(X, columns=self.feature_names_)

        result = []

        selected_rules = self.rules_
        for (r, _) in selected_rules:
            scores = np.zeros(X.shape[0], dtype=int)
            scores[np.array(range(X.shape[0]))[X.eval(r, engine='python')]] += 1
            profile = self._get_metrics(y, scores, average=average, describe=r)
            result.append(profile)
        desc = pd.DataFrame(result,
                            columns=['Lift_1', 'Precision_1', 'Recall_1', 'F1_score_1', 'Hit_1', 'Cnt_1', 'Lift_0',
                                     'Precision_0', 'Recall_0', 'F1_score_0', 'Hit_0', 'Cnt_0', 'Describe'])
        return desc.drop_duplicates()
