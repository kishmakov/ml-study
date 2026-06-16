import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.generator import (
    generate_ids,
    generate_sample_tensors,
    generate_samples,
    sample_restrictions,
)
from experiments.model import DeepSetPredictor

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_SAMPLES = 1 << 12

MIN_BITNESS = 4
MAX_BITNESS = 5

TRAIN_ITERATIONS = 10

REPS = 100
MODELS = {}
THRESHOLD = 0.05

PREDICT_BATCH_SIZE = 1024
TARGET_CASE_BATCH_SIZE = 128


def predict_values(model: nn.Module, x: np.ndarray) -> np.ndarray:
    model.to(DEVICE)
    model.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), PREDICT_BATCH_SIZE):
            xb = torch.as_tensor(
                x[start : start + PREDICT_BATCH_SIZE],
                dtype=torch.float32,
                device=DEVICE,
            )
            predictions.append(model(xb).cpu().numpy().ravel())


    return np.concatenate(predictions).astype(np.float32)


def approximate_targets(generator, bitness: int, case_ids: list[int]) -> np.ndarray:
    previous_model = MODELS.get(bitness - 1)
    if previous_model is None:
        raise ValueError(f"Missing model for bitness {bitness - 1}")

    target_parts = []
    ranges = range(0, len(case_ids), TARGET_CASE_BATCH_SIZE)
    total_batches = (len(case_ids) + TARGET_CASE_BATCH_SIZE - 1) // TARGET_CASE_BATCH_SIZE
    for start in tqdm(ranges, total=total_batches, desc=f"targets b={bitness}"):
        batch_ids = case_ids[start : start + TARGET_CASE_BATCH_SIZE]
        restricted_samples = []
        for case_id in batch_ids:
            restricted_samples.append(
                sample_restrictions(
                    generator,
                    bitness,
                    case_id,
                    REPS,
                )
            )

        x_restricted = np.concatenate(restricted_samples, axis=0)
        predictions = predict_values(previous_model, x_restricted)
        predictions = predictions.reshape(len(batch_ids), bitness, 2)
        branch_sizes = np.maximum(np.expm1(predictions), 0.0)
        split_sizes = 1.0 + branch_sizes.sum(axis=2)
        target_parts.append(np.log1p(split_sizes.min(axis=1)))

    return np.concatenate(target_parts).astype(np.float32)


def build_dataset(generator, bitness: int, seed: int):
    ids = generate_ids(generator, bitness, TRAIN_SAMPLES, seed)
    if bitness <= 4:
        x_train, y_train = generate_samples(generator, bitness, ids, REPS)
        y_train = np.log1p(y_train)
        return x_train, y_train

    x_train = generate_sample_tensors(generator, bitness, ids, REPS)
    y_train = approximate_targets(generator, bitness, ids)
    return x_train, y_train


def train(
    bitness: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
):
    model = MODELS.get(bitness)
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
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

        print(
            f"Epoch {epoch:>3}/{epochs}  "
            f"loss={epoch_loss:.4f}  "
            f"elapsed={epoch_elapsed:.2f}s  "
            f"device={DEVICE}"
        )

        if epoch_loss < THRESHOLD:
            break

def run_experiment(generator):
    SEED_OFFSET = 239

    for bitness in range(MIN_BITNESS, MAX_BITNESS + 1):
        point_dim = (bitness + 1) * (bitness + 1)
        MODELS[bitness] = DeepSetPredictor(point_dim)

    for iteration in range(TRAIN_ITERATIONS):
        for bitness in range(MIN_BITNESS, MAX_BITNESS + 1):
            seed = iteration + SEED_OFFSET
            x_train, y_train = build_dataset(generator, bitness, seed)
            train(bitness, x_train, y_train)


    # x_train, y_train, x_test, y_test = build_dataset(generator)
    # print(f"x_train: {x_train.shape}, y_train: {y_train.shape}")
    # print(f"x_test: {x_test.shape}, y_test: {y_test.shape}")
    # print(f"train labels: {np.bincount(y_train)}")
    # print(f"test labels: {np.bincount(y_test)}")
    #
    # train_detector(x_train, y_train, x_test, y_test)
