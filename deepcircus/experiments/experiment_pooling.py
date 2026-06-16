import ctypes
import multiprocessing as mp
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

from experiments.model import MLPDetector

TRAIN_SAMPLES_PER_SERIES = 1 << 11
TEST_SAMPLES_PER_SERIES = 1 << 11
RANDOM_SEED = 239
REPS = 100
PROCESSES = 16
POOL_CHUNKSIZE = 1
CACHE_DIR = Path(__file__).resolve().parents[1] / "tmp"
CACHE_FILES = {
    "x_train": CACHE_DIR / "x_train.npy",
    "y_train": CACHE_DIR / "y_train.npy",
    "x_test": CACHE_DIR / "x_test.npy",
    "y_test": CACHE_DIR / "y_test.npy",
}

_WORKER_GENERATOR = None
_WORKER_INPUT_BITNESS = 0


def append_bit(point: list[float], bit: bool):
    point.append(1.0 if bit else -1.0)


def append_bits(points, input_bits, generator, series_id, case_id):
    point = []
    output_bit = generator.generator_case_value(
        series_id,
        case_id,
        input_bits.encode("ascii"),
    )
    for bit in input_bits:
        append_bit(point, bit == "1")

    append_bit(point, output_bit)
    points.append(point)


def sample(series_id: int, case_id: int, generator, input_bitness: int):
    rng = random.Random((series_id << 32) + case_id)

    # num_vertices = generator.generator_case_nodes(series_id, case_id)
    # active_bits = generator.generator_case_active_bits(series_id, case_id).decode("ascii")

    # print(f"series={series_id} case={case_id} vertices={num_vertices} active={active_bits}")

    points = []
    for _ in range(REPS):
        input_bits = "".join(rng.choice("01") for _ in range(input_bitness))
        append_bits(points, input_bits, generator, series_id, case_id)

        flipped_bits = list(input_bits)
        for bit_id in range(input_bitness):
            flipped_bits[bit_id] = "0" if flipped_bits[bit_id] == "1" else "1"
            append_bits(points, "".join(flipped_bits), generator, series_id, case_id)
            flipped_bits[bit_id] = input_bits[bit_id]

    return np.asarray(points, dtype=np.float32).ravel()


def configure_worker_generator(generator):
    generator.generator_get_input_bitness.argtypes = []
    generator.generator_get_input_bitness.restype = ctypes.c_size_t

    generator.generator_get_series_number.argtypes = []
    generator.generator_get_series_number.restype = ctypes.c_size_t

    generator.generator_get_cases_number.argtypes = [ctypes.c_size_t]
    generator.generator_get_cases_number.restype = ctypes.c_size_t

    generator.generator_case_nodes.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    generator.generator_case_nodes.restype = ctypes.c_size_t

    generator.generator_case_active_bits.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    generator.generator_case_active_bits.restype = ctypes.c_char_p

    generator.generator_case_value.argtypes = [
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    generator.generator_case_value.restype = ctypes.c_bool
    return generator


def init_worker(generator_or_path, input_bitness: int):
    global _WORKER_GENERATOR, _WORKER_INPUT_BITNESS
    if isinstance(generator_or_path, (str, bytes)):
        _WORKER_GENERATOR = configure_worker_generator(ctypes.CDLL(generator_or_path))
    else:
        _WORKER_GENERATOR = generator_or_path
    _WORKER_INPUT_BITNESS = input_bitness


def sample_worker(task):
    row_id, series_id, case_id = task
    return (
        row_id,
        series_id,
        sample(series_id, case_id, _WORKER_GENERATOR, _WORKER_INPUT_BITNESS),
    )


def cache_exists() -> bool:
    return all(path.exists() for path in CACHE_FILES.values())


def load_cached_dataset():
    print(f"Loading cached dataset from {CACHE_DIR}")
    return tuple(np.load(CACHE_FILES[name]) for name in CACHE_FILES)


def save_cached_dataset(x_train, y_train, x_test, y_test) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_FILES["x_train"], x_train)
    np.save(CACHE_FILES["y_train"], y_train)
    np.save(CACHE_FILES["x_test"], x_test)
    np.save(CACHE_FILES["y_test"], y_test)


def build_split(generator, split_specs, split_name: str):
    input_bitness = int(generator.generator_get_input_bitness())
    feature_size = REPS * (input_bitness + 1) * (input_bitness + 1)
    total_samples = sum(len(case_ids) for _, case_ids in split_specs)
    x = np.empty((total_samples, feature_size), dtype=np.float32)
    y = np.empty(total_samples, dtype=np.int64)

    def tasks():
        row_id = 0
        max_cases = max(len(case_ids) for _, case_ids in split_specs)
        for case_index in range(max_cases):
            for series_id, case_ids in split_specs:
                if case_index < len(case_ids):
                    yield row_id, series_id, case_ids[case_index]
                    row_id += 1

    context = mp.get_context("fork")
    worker_generator = getattr(generator, "_name", None) or generator
    with context.Pool(
        processes=PROCESSES,
        initializer=init_worker,
        initargs=(worker_generator, input_bitness),
    ) as pool:
        print(
            f"Generating {total_samples} {split_name} feature vectors "
            f"with {PROCESSES} processes"
        )
        results = pool.imap_unordered(sample_worker, tasks(), chunksize=POOL_CHUNKSIZE)
        for row_id, series_id, features in tqdm(
            results,
            total=total_samples,
            desc=f"{split_name} features",
        ):
            x[row_id] = features
            y[row_id] = series_id

    return x, y


def build_dataset(generator):
    series_number = int(generator.generator_get_series_number())
    series_ids = list(range(series_number))
    train_specs = []
    test_specs = []

    for series_id in series_ids:
        cases_number = generator.generator_get_cases_number(series_id)
        needed_cases = TRAIN_SAMPLES_PER_SERIES + TEST_SAMPLES_PER_SERIES
        assert needed_cases < cases_number, "Asking too much"

        rng = random.Random(RANDOM_SEED + series_id)
        case_ids = rng.sample(range(cases_number), needed_cases)
        train_specs.append((series_id, case_ids[:TRAIN_SAMPLES_PER_SERIES]))
        test_specs.append((series_id, case_ids[TRAIN_SAMPLES_PER_SERIES:]))

    x_train, y_train = build_split(
        generator,
        train_specs,
        "train",
    )
    x_test, y_test = build_split(
        generator,
        test_specs,
        "test",
    )

    # save_cached_dataset(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test


def train(
    model: MLPDetector,
    x_train: np.ndarray,
    y_train: np.ndarray,
    device: str,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> MLPDetector:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
    )

    x = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(x)
        scheduler.step(epoch_loss)
        epoch_elapsed = time.perf_counter() - epoch_start

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"loss={epoch_loss:.4f}  "
                f"elapsed={epoch_elapsed:.2f}s  "
                f"device={device}"
            )

    return model


def predict(model: MLPDetector, x_test: np.ndarray, *, device: str = "cpu") -> np.ndarray:
    model.eval()
    x = torch.tensor(x_test, dtype=torch.float32)
    predictions = []

    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(x), batch_size=1024):
            logits = model(xb.to(device))
            batch_predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
            predictions.append(batch_predictions.cpu().numpy().ravel())

    return np.concatenate(predictions)


def train_detector(x_train, y_train, x_test, y_test):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(RANDOM_SEED)
    model = MLPDetector(input_dim=x_train.shape[1])
    model = train(model, x_train, y_train, device)
    predictions = predict(model, x_test, device=device)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)

    print(f"detector_test_accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"baseline_test_accuracy: {accuracy_score(y_test, baseline_predictions):.4f}")
    print(classification_report(y_test, predictions))


def run_experiment(generator):
    if cache_exists():
        x_train, y_train, x_test, y_test = load_cached_dataset()
    else:
        x_train, y_train, x_test, y_test = build_dataset(generator)

    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
    print(f"train labels: {np.bincount(y_train)}")
    print(f"test labels: {np.bincount(y_test)}")

    train_detector(x_train, y_train, x_test, y_test)
