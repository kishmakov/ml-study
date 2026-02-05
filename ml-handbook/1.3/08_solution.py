import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin

class RubricCityMedianClassifier(ClassifierMixin):
    def fit(self, X=None, y=None):
        U = X.copy()
        U["_y"] = y

        self.mean_bill = (
            U
            .groupby(["city", "modified_rubrics"])["_y"]
            .median()
            .to_dict()
        )
            
        return self

    def predict(self, X=None):
        return np.array([
            self.mean_bill.get((row.city, row.modified_rubrics), 1.0)
            for row in X.itertuples()
        ])