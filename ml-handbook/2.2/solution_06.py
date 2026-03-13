import numpy as np
from sklearn.base import RegressorMixin

class SGDLinearRegressor(RegressorMixin):
    def __init__(self,
                 lr=0.01, regularization=1., delta_converged=1e-3, max_steps=1000,
                 batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        self.W = np.random.randn(X.shape[1]) * 0.01
        self.b = 0

        self.steps = 0

        for epoch in range(self.max_steps):
            for start in range(0, X.shape[1], self.batch_size):
                end = start + self.batch_size

                X_batch = X[start:end]
                Y_batch = Y[start:end]

                lin = X_batch @ self.W + self.b - Y_batch

                dW = 2 * X_batch.T @ lin / self.batch_size + 2 * self.regularization * self.W
                db = 2 * np.sum(lin) / self.batch_size

                dW = dW * self.lr
                db = db * self.lr

                self.pb = self.b
                self.pW = self.W

                self.W = self.W + dW
                self.b = self.b + db

                self.steps += 1
                dL2 = np.sqrt(np.sum(dW ** 2))

                if (self.steps > self.max_steps) or dL2 < self.delta_converged:
                    return self


        return self

    def predict(self, X):
        return X @ self.W + self.b