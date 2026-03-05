import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import Ridge


class ExponentialLinearRegression(RegressorMixin):
    def __init__(self, *args, **kwargs):
        self.ridge = Ridge(*args, **kwargs)

    def fit(self, X, Y):
        self.ridge.fit(X, np.log(Y))
        return self

    def predict(self, X):
        return np.exp(self.ridge.predict(X))

    def get_params(self, deep=True):
        return self.ridge.get_params(deep)

    def set_params(self, **params):
        self.ridge.set_params(**params)
        return self