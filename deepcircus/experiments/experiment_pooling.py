import hashlib
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

TRAIN_SAMPLES_PER_BITNESS = 1 << 11
TEST_SAMPLES_PER_BITNESS = 1 << 11
RANDOM_SEED = 239
REPS = 100
PROCESSES = 16
POOL_CHUNKSIZE = 1
GENERATION_BITNESSES = (
    4,
    15,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


_WORKER_GENERATOR = None
_WORKER_INPUT_BITNESS = 0


def append_bit(point: list[float], bit: bool):
    point.append(1.0 if bit else -1.0)


def append_bits(points, input_bits, generator, bitness: int, case_id: int):
    point = []
    output_bit = generator.case_value(
        bitness,
        case_id,
        input_bits[:bitness],
    )
    for bit in input_bits:
        append_bit(point, bit == "1")

    append_bit(point, output_bit)
    points.append(point)


def sample(bitness: int, case_id: int, generator, input_bitness: int):
    rng = random.Random((bitness << 32) + case_id)

    points = []
    for _ in range(REPS):
        input_bits = "".join(rng.choice("01") for _ in range(input_bitness))
        append_bits(points, input_bits, generator, bitness, case_id)

        flipped_bits = list(input_bits)
        for bit_id in range(input_bitness):
            flipped_bits[bit_id] = "0" if flipped_bits[bit_id] == "1" else "1"
            append_bits(points, "".join(flipped_bits), generator, bitness, case_id)
            flipped_bits[bit_id] = input_bits[bit_id]

    return np.asarray(points, dtype=np.float32).ravel()


def init_worker(generator, input_bitness: int):
    global _WORKER_GENERATOR, _WORKER_INPUT_BITNESS
    _WORKER_GENERATOR = generator
    _WORKER_INPUT_BITNESS = input_bitness


def sample_worker(task):
    row_id, bitness_id, bitness, case_id = task
    return (
        row_id,
        bitness_id,
        sample(bitness, case_id, _WORKER_GENERATOR, _WORKER_INPUT_BITNESS),
    )


def build_split(generator, split_specs, split_name: str):
    input_bitness = max(GENERATION_BITNESSES)
    feature_size = REPS * (input_bitness + 1) * (input_bitness + 1)
    total_samples = sum(len(case_ids) for _, _, case_ids in split_specs)
    x = np.empty((total_samples, feature_size), dtype=np.float32)
    y = np.empty(total_samples, dtype=np.int64)

    def tasks():
        row_id = 0
        max_cases = max(len(case_ids) for _, _, case_ids in split_specs)
        for case_index in range(max_cases):
            for bitness_id, bitness, case_ids in split_specs:
                if case_index < len(case_ids):
                    yield row_id, bitness_id, bitness, case_ids[case_index]
                    row_id += 1

    context = mp.get_context("fork")
    with context.Pool(
        processes=PROCESSES,
        initializer=init_worker,
        initargs=(generator, input_bitness),
    ) as pool:
        print(
            f"Generating {total_samples} {split_name} feature vectors "
            f"with {PROCESSES} processes"
        )
        results = pool.imap_unordered(sample_worker, tasks(), chunksize=POOL_CHUNKSIZE)
        for row_id, bitness_id, features in tqdm(
            results,
            total=total_samples,
            desc=f"{split_name} features",
        ):
            x[row_id] = features
            y[row_id] = bitness_id

    return x, y


def build_dataset(generator):
    train_specs = []
    test_specs = []

    for bitness_id, bitness in enumerate(GENERATION_BITNESSES):
        cases_number = generator.cases_number(bitness)
        needed_cases = TRAIN_SAMPLES_PER_BITNESS + TEST_SAMPLES_PER_BITNESS
        assert needed_cases < cases_number, "Asking too much"

        rng = random.Random(RANDOM_SEED + bitness)
        case_ids = rng.sample(range(cases_number), needed_cases)
        train_specs.append((bitness_id, bitness, case_ids[:TRAIN_SAMPLES_PER_BITNESS]))
        test_specs.append((bitness_id, bitness, case_ids[TRAIN_SAMPLES_PER_BITNESS:]))

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

    return x_train, y_train, x_test, y_test


def train(
    model: MLPDetector,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> MLPDetector:
    model.to(DEVICE)
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
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
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
                f"device={DEVICE}"
            )

    return model


def predict(model: MLPDetector, x_test: np.ndarray) -> np.ndarray:
    model.eval()
    x = torch.tensor(x_test, dtype=torch.float32)
    predictions = []

    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(x), batch_size=1024):
            logits = model(xb.to(DEVICE))
            batch_predictions = (torch.sigmoid(logits) >= 0.5).to(torch.int64)
            predictions.append(batch_predictions.cpu().numpy().ravel())

    return np.concatenate(predictions)


def train_detector(x_train, y_train, x_test, y_test):
    torch.manual_seed(RANDOM_SEED)
    model = MLPDetector(input_dim=x_train.shape[1])
    model = train(model, x_train, y_train)
    predictions = predict(model, x_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    baseline_predictions = baseline.predict(x_test)

    print(f"detector_test_accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"baseline_test_accuracy: {accuracy_score(y_test, baseline_predictions):.4f}")
    print(classification_report(y_test, predictions))


def run_experiment(generator):
    x_train, y_train, x_test, y_test = build_dataset(generator)
    print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
    print(f"train labels: {np.bincount(y_train)}")
    print(f"test labels: {np.bincount(y_test)}")

    train_detector(x_train, y_train, x_test, y_test)
