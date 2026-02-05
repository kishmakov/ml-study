
import numpy as np

from sklearn.base import RegressorMixin

class MeanRegressor(RegressorMixin):
    # Predicts the mean of y_train
    def fit(self, X=None, y=None):
        self.y_mean = np.mean(y)
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        return self

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        return self.y_mean
    
estimator = MeanRegressor()
X = np.array([[1, 2], [2, 3], [3, 4]])
y = np.array([-1, 0, 1])
print("Estimeate: ", estimator.fit(X, y).predict(X))