import numpy as np
from sklearn.base import RegressorMixin

class SGDLinearRegressor(RegressorMixin):
    def __init__(self,
                 lr=0.01, regularization=1.0, delta_converged=1e-3, max_steps=1000,
                 batch_size=64):
        self.lr = lr
        self.regularization = regularization
        self.max_steps = max_steps
        self.delta_converged = delta_converged
        self.batch_size = batch_size

        self.W = None
        self.b = None

    def fit(self, X, Y):
        self.W = np.random.uniform(-1.0, 1.0, size=X.shape[1])
        self.b = 0

        for _ in range(self.max_steps):
            ids = np.random.randint(0, X.shape[0], size=self.batch_size)
            batch_n = len(ids)

            X_batch = X[ids]
            Y_batch = Y[ids]

            lin = X_batch @ self.W + self.b - Y_batch

            dW = 2 * X_batch.T @ lin / batch_n + 2 * self.regularization * self.W
            db = 2 * np.sum(lin) / batch_n

            dL2 = np.sqrt(np.sum(dW ** 2))

            dW = dW * self.lr
            db = db * self.lr

            self.W -= dW
            self.b -= db

            if dL2 < self.delta_converged:
                break

        return self

    def predict(self, X):
        return X @ self.W + self.b