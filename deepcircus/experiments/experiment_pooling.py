import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

SAMPLE_SIZE = 1 << 10
TRAIN_FRACTION = 0.8
RANDOM_SEED = 239
REPS = 5
SERIES_ID = 0


def append_bit(point: list[float], bit: bool):
    point.append(1.0 if bit else -1.0)


def append_bits(points, input_bits, generator, case_id):
    point = []
    output_bit = generator.generator_case_value(
        SERIES_ID,
        case_id,
        input_bits.encode("ascii"),
    )
    for bit in input_bits:
        append_bit(point, bit == "1")

    append_bit(point, output_bit)
    points.append(point)


def sample(case_id: int, generator):
    input_bitness = generator.generator_get_input_bitness()
    rng = random.Random(case_id)

    points = []
    for _ in range(REPS):
        input_bits = "".join(rng.choice("01") for _ in range(input_bitness))
        append_bits(points, input_bits, generator, case_id)

        flipped_bits = list(input_bits)
        for bit_id in range(input_bitness):
            flipped_bits[bit_id] = "0" if flipped_bits[bit_id] == "1" else "1"
            append_bits(points, "".join(flipped_bits), generator, case_id)
            flipped_bits[bit_id] = input_bits[bit_id]

    return {
        "X": np.asarray(points, dtype=np.float32).ravel(),
        "Y": generator.generator_case_nodes(SERIES_ID, case_id),
    }


def train_simple(train_df, test_df):
    x_train = np.vstack(train_df["X"])
    y_train = np.log1p(train_df["Y"].to_numpy())
    x_test = np.vstack(test_df["X"])
    y_test = np.log1p(test_df["Y"].to_numpy())

    # model = Ridge(max_iter=1000, random_state=RANDOM_SEED)
    model = LinearRegression()
    model.fit(x_train, y_train)

    logreg_predictions = model.predict(x_test)
    baseline_predictions = np.full_like(y_test, y_train.mean(), dtype=np.float64)

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
        for case_id in tqdm(case_ids)
    ]

    df = pd.DataFrame(rows)

    train_size = int(len(df) * TRAIN_FRACTION)

    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    train_avg_nodes = train_df["Y"].mean()
    test_avg_nodes = test_df["Y"].mean()


    print(f"train_avg_nodes: {train_avg_nodes:.2f}")
    print(f"test_avg_nodes: {test_avg_nodes:.2f}")
    train_simple(train_df, test_df)

    return df
