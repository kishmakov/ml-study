import numpy as np

def root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.):
    if np.any(y_true <= 0.0):
        raise ValueError("y_true has non-positive value")

    y_pred = np.maximum(y_pred, a_min)
    diff = np.log(y_true) - np.log(y_pred)
    return np.sqrt(np.mean(diff ** 2))

