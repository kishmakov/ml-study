import numpy as np

from sklearn.base import ClassifierMixin

class MostFrequentClassifier(ClassifierMixin):
    # Predicts the rounded (just in case) median of y_train
    def fit(self, X=None, y=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Training data features
        y : array like, shape = (_samples,)
        Training data targets
        '''
        values, counts = np.unique(y, return_counts=True)        
        self.most_frequent = values[np.argmax(counts)]
        return self

    def predict(self, X=None):
        '''
        Parameters
        ----------
        X : array like, shape = (n_samples, n_features)
        Data to predict
        '''
        return np.full(X.shape[0], self.most_frequent)
    
estimator = MostFrequentClassifier()
X = np.array([[1, 2], [2, 3], [3, 4], [5, 1], [0, 1]])
y = np.array([1, 2, 1, 2, 2])
print("Estimeate: ", estimator.fit(X, y).predict(X))