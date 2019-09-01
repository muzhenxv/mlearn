from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LogisticRegressionEstimator(LogisticRegression):
    # def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
    #              fit_intercept=True, intercept_scaling=1, class_weight=None,
    #              random_state=None, solver='liblinear', max_iter=100,
    #              multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):
    #     super(LogisticRegressionEstimator, self).__init__(penalty=penalty,
    #         dual=dual, tol=tol, C=C, fit_intercept=fit_intercept,
    #         intercept_scaling=intercept_scaling, class_weight=class_weight,
    #         random_state=random_state, solver=solver, max_iter=max_iter,
    #         multi_class=multi_class, verbose=verbose, warm_start=warm_start,
    #         n_jobs=n_jobs)
    #
    #     self.std_scaler = StandardScaler()
    def __init__(self, std_scaler=None, **kwargs):
        super(LogisticRegressionEstimator, self).__init__(**kwargs)

        if std_scaler is None:
            self.std_scaler = StandardScaler()

    def fit(self, X, y, sample_weight=None):
        X_scal = self.std_scaler.fit_transform(X)
        super().fit(X_scal, y, sample_weight=sample_weight)
        return self
        
    def transform(self, X):
        X_scal = self.std_scaler.transform(X)
        return super().transform(X_scal)
    
    def predict(self, X):
        X_scal = self.std_scaler.transform(X)
        return super().predict(X_scal)
        
    def predict_proba(self, X):
        X_scal = self.std_scaler.transform(X)
        return super().predict_proba(X_scal)
    