import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report

from experiments.generator import generate_samples
from experiments.model import MLPDetector

TRAIN_SAMPLES_PER_BITNESS = 1 << 10
TEST_SAMPLES_PER_BITNESS = 1 << 10
RANDOM_SEED = 239
REPS = 100

GENERATION_BITNESSES = (4, 15)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def pad_samples(x: np.ndarray, bitness: int, target_bitness: int) -> np.ndarray:
    if bitness == target_bitness:
        return x

    source_side = bitness + 1
    target_side = target_bitness + 1
    reshaped = x.reshape(len(x), REPS, source_side, source_side)
    padded = np.zeros((len(x), REPS, target_side, target_side), dtype=np.float32)
    padded[:, :, :source_side, :source_side] = reshaped
    return padded.reshape(len(x), REPS * target_side * target_side)


def build_dataset(generator):
    target_bitness = max(GENERATION_BITNESSES)
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

    x_train_parts = []
    y_train_parts = []
    for bitness_id, bitness, case_ids in train_specs:
        x_train_parts.append(
            pad_samples(
                generate_samples(
                    generator,
                    bitness,
                    case_ids=case_ids,
                    reps=REPS,
                    split_name="train",
                ),
                bitness,
                target_bitness,
            )
        )
        y_train_parts.append(np.full(len(case_ids), bitness_id, dtype=np.int64))

    x_test_parts = []
    y_test_parts = []
    for bitness_id, bitness, case_ids in test_specs:
        x_test_parts.append(
            pad_samples(
                generate_samples(
                    generator,
                    bitness,
                    case_ids=case_ids,
                    reps=REPS,
                    split_name="test",
                ),
                bitness,
                target_bitness,
            )
        )
        y_test_parts.append(np.full(len(case_ids), bitness_id, dtype=np.int64))

    x_train = np.concatenate(x_train_parts, axis=0)
    y_train = np.concatenate(y_train_parts, axis=0)
    x_test = np.concatenate(x_test_parts, axis=0)
    y_test = np.concatenate(y_test_parts, axis=0)

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
