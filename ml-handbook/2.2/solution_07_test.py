import numpy as np
import pandas as pd
import pytest
from solution_07 import (
    BaseDataPreprocessor,
    OneHotPreprocessor,
    make_ultimate_pipeline,
    continuous_columns,
    interesting_columns,
)

N_ROWS = 50
RNG = np.random.default_rng(42)


@pytest.fixture
def sample_df():
    data = {col: RNG.uniform(0, 100, N_ROWS) for col in continuous_columns}
    data["Overall_Qual"] = RNG.choice(["Ex", "Gd", "TA", "Fa"], N_ROWS)
    data["Garage_Qual"] = RNG.choice(["Ex", "Gd", "TA", "Fa", "Po"], N_ROWS)
    data["Sale_Condition"] = RNG.choice(["Normal", "Abnorml", "Partial"], N_ROWS)
    data["MS_Zoning"] = RNG.choice(["RL", "RM", "C (all)"], N_ROWS)
    return pd.DataFrame(data)


@pytest.fixture
def target(sample_df):
    return RNG.uniform(50_000, 500_000, len(sample_df))


# --- BaseDataPreprocessor ---

def test_base_preprocessor_fit_returns_self(sample_df):
    pre = BaseDataPreprocessor()
    assert pre.fit(sample_df) is pre


def test_base_preprocessor_transform_shape(sample_df):
    pre = BaseDataPreprocessor()
    pre.fit(sample_df)
    result = pre.transform(sample_df)
    assert result.shape == (N_ROWS, len(continuous_columns))


def test_base_preprocessor_transform_scaled(sample_df):
    pre = BaseDataPreprocessor()
    pre.fit(sample_df)
    result = pre.transform(sample_df)
    # StandardScaler: mean ~0, std ~1 per column
    assert np.allclose(result.mean(axis=0), 0, atol=1e-10)
    assert np.allclose(result.std(axis=0), 1, atol=1e-10)


def test_base_preprocessor_none_columns(sample_df):
    # When needed_columns=None all columns are passed to StandardScaler,
    # which cannot handle string categoricals — document the known limitation.
    numeric_df = sample_df[continuous_columns]
    pre = BaseDataPreprocessor(needed_columns=None)
    pre.fit(numeric_df)
    result = pre.transform(numeric_df)
    assert result.shape == (N_ROWS, len(continuous_columns))


def test_base_preprocessor_custom_columns(sample_df):
    cols = ["Lot_Area", "Gr_Liv_Area", "Year_Built"]
    pre = BaseDataPreprocessor(needed_columns=cols)
    pre.fit(sample_df)
    result = pre.transform(sample_df)
    assert result.shape == (N_ROWS, len(cols))


# --- OneHotPreprocessor ---

def test_one_hot_preprocessor_fit_returns_self(sample_df):
    pre = OneHotPreprocessor()
    assert pre.fit(sample_df) is pre


def test_one_hot_preprocessor_transform_shape(sample_df):
    pre = OneHotPreprocessor()
    pre.fit(sample_df)
    result = pre.transform(sample_df)
    # continuous columns + one-hot encoded columns
    n_oh_features = sum(
        len(sample_df[col].unique()) for col in interesting_columns
    )
    assert result.shape == (N_ROWS, len(continuous_columns) + n_oh_features)


def test_one_hot_preprocessor_unknown_category(sample_df):
    pre = OneHotPreprocessor()
    pre.fit(sample_df)
    # Introduce an unseen category; handle_unknown="ignore" should not raise
    test_df = sample_df.copy()
    test_df.loc[0, "MS_Zoning"] = "UNKNOWN_ZONE"
    result = pre.transform(test_df)
    assert result.shape[0] == N_ROWS


# --- Pipeline ---

def test_pipeline_fit_predict(sample_df, target):
    pipeline = make_ultimate_pipeline()
    pipeline.fit(sample_df, target)
    predictions = pipeline.predict(sample_df)
    assert predictions.shape == (N_ROWS,)
    assert np.all(np.isfinite(predictions))


def test_pipeline_fit_predict_on_subset(sample_df, target):
    pipeline = make_ultimate_pipeline()
    train, test = sample_df.iloc[:40], sample_df.iloc[40:]
    pipeline.fit(train, target[:40])
    predictions = pipeline.predict(test)
    assert predictions.shape == (10,)
    assert np.all(np.isfinite(predictions))
