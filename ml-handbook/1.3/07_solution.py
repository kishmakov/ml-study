import pandas as pd

from sklearn.base import RegressorMixin

class CityMeanRegressor(RegressorMixin):
    def fit(self, X=None, y=None):
        y = pd.Series(y, index=X.index, name="average_bill")
        self.mean_bill = y.groupby(X["city"]).mean()
        return self

    def predict(self, X=None):
        return X["city"].map(self.mean_bill)  