import random
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

SAMPLE_SIZE = 1 << 15
TRAIN_FRACTION = 0.8
RANDOM_SEED = 239
REPS = 5
SERIES_ID = 0


def sample(case_id: int, generator):
    input_bitness = generator.generator_get_input_bitness()
    rng = random.Random(case_id)

    points = np.empty((REPS, input_bitness + 1), dtype=np.float32)
    for point_id in range(REPS):
        input_bits = "".join(rng.choice("01") for _ in range(input_bitness))
        output_bit = generator.generator_case_value(
            SERIES_ID,
            case_id,
            input_bits.encode("ascii"),
        )
        points[point_id, :-1] = np.fromiter(
            (1 if bit == "1" else -1 for bit in input_bits),
            dtype=np.float32,
            count=input_bitness,
        )
        points[point_id, -1] = 1.0 if output_bit else -1.0

    return {
        "X": points.ravel(),
        "Y": generator.generator_case_nodes(SERIES_ID, case_id),
    }


def build_feature_matrix(df):
    return np.vstack(df["X"])


def train_logreg(train_df, test_df):
    x_train = build_feature_matrix(train_df)
    y_train = train_df["Y"].to_numpy()
    x_test = build_feature_matrix(test_df)
    y_test = test_df["Y"].to_numpy()

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(x_train, y_train)

    logreg_predictions = model.predict(x_test)
    baseline_predictions = np.full_like(y_test, y_test.mean(), dtype=np.float64)

    print(f"logreg_test_mae: {mean_absolute_error(y_test, logreg_predictions):.2f}")
    print(f"logreg_test_mse: {mean_squared_error(y_test, logreg_predictions):.2f}")
    print(f"baseline_test_mae: {mean_absolute_error(y_test, baseline_predictions):.2f}")
    print(f"baseline_test_mse: {mean_squared_error(y_test, baseline_predictions):.2f}")


def run_pooling_experiment(generator):
    cases_number = generator.generator_get_cases_number(SERIES_ID)
    sample_size = min(SAMPLE_SIZE, cases_number)

    rng = random.Random(RANDOM_SEED)
    case_ids = rng.sample(range(cases_number), sample_size)

    rows = [
        sample(case_id, generator)
        for case_id in case_ids
    ]

    df = pd.DataFrame(rows)

    train_size = int(len(df) * TRAIN_FRACTION)

    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    train_avg_nodes = train_df["Y"].mean()
    test_avg_nodes = test_df["Y"].mean()


    print(f"train_avg_nodes: {train_avg_nodes:.2f}")
    print(f"test_avg_nodes: {test_avg_nodes:.2f}")
    train_logreg(train_df, test_df)

    return df
