from solution_02 import root_mean_squared_logarithmic_error
import pytest


def reference_rmsle(y_true, y_pred, a_min=1.):
    y_pred = np.maximum(y_pred, a_min)
    diff = np.log(y_true) - np.log(y_pred)
    return np.sqrt(np.mean(diff ** 2))


def test_basic_case():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1, 2, 3, 4])
    assert root_mean_squared_logarithmic_error(y_true, y_pred) == 0.0


def test_known_value():
    y_true = np.array([1, 2, 4])
    y_pred = np.array([1, 3, 4])

    expected = reference_rmsle(y_true, y_pred)
    result = root_mean_squared_logarithmic_error(y_true, y_pred)

    assert np.isclose(result, expected)


def test_prediction_clipping():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([-10, 0.1, 0.5])

    result = root_mean_squared_logarithmic_error(y_true, y_pred, a_min=1.0)

    clipped = np.maximum(y_pred, 1.0)
    expected = np.sqrt(np.mean((np.log(y_true) - np.log(clipped)) ** 2))

    assert np.isclose(result, expected)


def test_non_positive_y_true():
    y_true = np.array([1, 0, 3])
    y_pred = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        root_mean_squared_logarithmic_error(y_true, y_pred)


def test_random_vectors():
    rng = np.random.default_rng(0)
    y_true = rng.uniform(1, 100, 1000)
    y_pred = rng.uniform(-50, 100, 1000)

    expected = reference_rmsle(y_true, y_pred)
    result = root_mean_squared_logarithmic_error(y_true, y_pred)

    assert np.isclose(result, expected)