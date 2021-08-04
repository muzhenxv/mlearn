import pandas as pd
import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin
from decimal import Decimal

class Psi(BaseEstimator, TransformerMixin):
    def __init__(self, q=10, max_th=10, min_th=-10, smooth=0.0001, range_ratio=0.2, precision=8):
        self.max_th = max_th
        self.min_th = min_th
        self.smooth = smooth
        self.range_ratio = range_ratio
        self.q = q
        self.precision = precision

    def fit(self, expect_score):
        expect_score = pd.Series(expect_score)

        self.bins = self.qcut_duplicates(expect_score, q=self.q, range_ratio=self.range_ratio, precision=self.precision)
        self.expect_dist = pd.DataFrame(
            pd.cut(expect_score, bins=self.bins, include_lowest=True).value_counts()).reset_index()
        self.expect_dist.columns = ['cat', 'expect_score']
        return self

    def transform(self, actual_score):
        actual_score = pd.Series(actual_score)

        s = pd.cut(actual_score, bins=self.bins, include_lowest=True).value_counts() + self.smooth
        s.columns = ['actual_score']
        extreme_ratio = (len(actual_score) + float(len(s) * Decimal(str(self.smooth))) - float(
            Decimal(str(s.sum())).quantize(Decimal(str(self.smooth))))) / len(actual_score)

        self.dist = self.expect_dist.copy()
        self.dist['actual_score'] = self.dist.cat.map(s).astype(float)

        self.dist['expect_score_ratio'] = self.dist.expect_score / self.dist.expect_score.sum()
        self.dist['actual_score_ratio'] = self.dist.actual_score / self.dist.actual_score.sum()

        lg = np.log(self.dist.actual_score_ratio / self.dist.expect_score_ratio)
        lg[lg > self.max_th] = self.max_th
        lg[lg < self.min_th] = self.min_th
        p = np.sum((self.dist.actual_score_ratio - self.dist.expect_score_ratio) * lg)

        return p, extreme_ratio

    @staticmethod
    def qcut(l, q=10, range_ratio=0.1, precision=8):
        _, cut_points = pd.qcut(l, q=q, retbins=True, precision=precision, duplicates='drop')

        span = cut_points[-1] - cut_points[0]
        cut_points[0] -= span * range_ratio
        cut_points[-1] += span * range_ratio

        return cut_points
    
    @staticmethod
    def qcut_duplicates(X, bins=10, range_ratio=0.1, precision=8, duplicates='drop', **kwargs):
        """
        用于消除此种情况：即使取值数很多，在某几列的占比非常高的情况下，即使要求分箱数为10，最终分箱结果可能依旧只有少数几类。
        """
        X_copy = pd.Series(X)
        t = X_copy.value_counts(normalize=True)
        points1 = t[t>=0.5/bins].index.tolist()
        X_copy = X_copy[~X_copy.isin(points1)]
        _, points2 = pd.qcut(X_copy, q=bins-len(points1), retbins=True, duplicates=duplicates, precision=precision)
        cut_points = sorted(points1 + list(points2))
        span = cut_points[-1] - cut_points[0]
        cut_points[0] -= span * range_ratio
        cut_points[-1] += span * range_ratio
        return cut_points


def cmpt_psi(x1, x2, smooth=0.01):
    x1_c = np.array(list(x1)) + smooth
    x2_c = np.array(list(x2)) + smooth
    return sum((np.array(x1_c) - np.array(x2_c)) * np.log((np.array(x1_c) / np.array(x2_c))))


if __name__ == '__main__':
    expect_data = [random.uniform(0, 1) for i in range(1000)]
    actual_data = [random.uniform(0, 1) for i in range(3000)]
    expect_data = [random.randint(0, 2) for i in range(1000)]
    actual_data = [random.randint(-1, 2) for i in range(3000)]
    psi = Psi(range_ratio=0.1)
    psi.fit(expect_data)
    print(psi.transform(actual_data))
