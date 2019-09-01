from skrules import SkopeRules
import numpy as np
from ..base_utils import RulerMixin


class SkopeRuler(SkopeRules, RulerMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.random_state is None:
            self.random_state = 7

    def fit(self, X, y, sample_weight=None):
        self.feature_names = list(X.columns)
        super().fit(X, y, sample_weight=sample_weight)
        return self

    def predict_proba(self, X):
        a = self.score_top_rules(X)
        t = np.ones((len(a), 2))
        t[:, 1] = a
        t[:, 0] = -a
        return t
