import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from experiments.sampler import (
    generate_ids,
    generate_restriction_tensors,
    generate_sample_tensors,
    generate_samples,
)
from experiments.model import DeepSetPredictor
from experiments.state_io import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_DIR,
    ensure_model_dir,
    initial_progress,
    load_model_checkpoint_if_exists,
    load_or_create_experiment_meta,
    save_completed_target_state,
    save_experiment_complete_state,
    save_experiment_meta,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = DEFAULT_MODEL_DIR
META_PATH = DEFAULT_META_PATH

TRAIN_SAMPLES = 1 << 12
VALIDATION_SAMPLES = 128

MIN_BITNESS = 4
MAX_BITNESS = 10

TRAIN_ITERATIONS = 100
TRAIN_EPOCHS = 500
BATCH_SIZE = 256
LR = 1e-3

REPS = 128
MODELS = {}
THRESHOLD = 0.025

PREDICT_BATCH_SIZE = 1024
TARGET_CASE_BATCH_SIZE = 128
TARGET_PROCESSES = 16
SEED_OFFSET = 239
VALIDATION_SEED_OFFSET = 1_000_239


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
    assert previous_model, f"Missing model for bitness {bitness - 1}"

    target_parts = []
    ranges = range(0, len(case_ids), TARGET_CASE_BATCH_SIZE)
    total_batches = (len(case_ids) + TARGET_CASE_BATCH_SIZE - 1) // TARGET_CASE_BATCH_SIZE

    print(
        f"Generating reduced samples for {len(case_ids)} targets "
        f"at bitness {bitness} with {TARGET_PROCESSES} processes"
    )
    for start in tqdm(ranges, total=total_batches, desc=f"targets b={bitness}"):
        batch_ids = case_ids[start : start + TARGET_CASE_BATCH_SIZE]
        x_restricted = generate_restriction_tensors(
            generator,
            bitness,
            batch_ids,
            REPS,
            TARGET_PROCESSES,
        )
        predictions = predict_values(previous_model, x_restricted)
        predictions = predictions.reshape(len(batch_ids), bitness, 2)
        predictions = np.clip(
            predictions,
            0.0,
            np.log1p((1 << bitness) - 1),
        )
        branch_sizes = np.maximum(np.expm1(predictions), 0.0)
        split_sizes = 1.0 + branch_sizes.sum(axis=2)
        target_parts.append(np.log1p(split_sizes.min(axis=1)))

    return np.concatenate(target_parts).astype(np.float32)


def build_dataset(generator, bitness: int, seed: int):
    ids = generate_ids(generator, bitness, TRAIN_SAMPLES, seed)
    if bitness <= 4:
        x_train, y_train = generate_samples(
            generator, bitness, ids, REPS, TARGET_PROCESSES
        )
        y_train = np.log1p(y_train)
        return x_train, y_train

    x_train = generate_sample_tensors(generator, bitness, ids, REPS, TARGET_PROCESSES)
    y_train = approximate_targets(generator, bitness, ids)
    return x_train, y_train


def evaluate_validation(
    model: nn.Module,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> dict[str, float]:
    predictions = predict_values(model, x_validation)
    errors = predictions - y_validation
    return {
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mae": float(np.mean(np.abs(errors))),
    }


def train(
    bitness: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = TRAIN_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> float:
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

    final_loss = float("inf")
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
        final_loss = float(epoch_loss)

        if epoch_loss < THRESHOLD or epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"loss={epoch_loss:.4f}  "
                f"elapsed={epoch_elapsed:.2f}s  "
                f"device={DEVICE}"
            )

        if epoch_loss < THRESHOLD:
            break

    return final_loss


def run_experiment(generator):
    ensure_model_dir(MODEL_DIR)
    config = build_config()
    meta = load_or_create_experiment_meta(
        META_PATH,
        config,
        initial_progress(config),
        create_validation_config,
    )
    progress = meta["progress"]
    if progress["stage"] == "done":
        print(f"training already complete: {META_PATH}")
        return

    validation = {}
    for bitness in range(MIN_BITNESS, MAX_BITNESS + 1):
        point_dim = 2 * bitness + 1
        MODELS[bitness] = DeepSetPredictor(point_dim)
        checkpoint = load_model_checkpoint_if_exists(MODEL_DIR, bitness, DEVICE)
        if checkpoint is not None:
            MODELS[bitness].load_state_dict(checkpoint["state_dict"])

    for iteration in range(int(progress["iteration"]), TRAIN_ITERATIONS):
        for bitness in range(MIN_BITNESS, MAX_BITNESS + 1):
            progress = meta["progress"]
            if iteration == progress["iteration"] and bitness < progress["bitness"]:
                continue

            seed = iteration + SEED_OFFSET
            x_train, y_train = build_dataset(generator, bitness, seed)
            loss = train(
                bitness,
                x_train,
                y_train,
            )
            validation[bitness] = validation.get(bitness) or load_validation_dataset(
                generator,
                meta,
                bitness,
            )
            validation_result = evaluate_validation(
                MODELS[bitness],
                validation[bitness]["x"],
                validation[bitness]["y"],
            )
            save_completed_target_state(
                meta,
                META_PATH,
                MODELS[bitness],
                config,
                MODEL_DIR,
                bitness,
                iteration,
                loss,
                validation_result,
            )

    save_experiment_complete_state(meta, META_PATH, config)


def build_config() -> dict:
    return {
        "model_dir": MODEL_DIR,
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "min_bitness": MIN_BITNESS,
        "max_bitness": MAX_BITNESS,
        "train_iterations": TRAIN_ITERATIONS,
        "train_epochs": TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "reps": REPS,
        "threshold": THRESHOLD,
        "seed_offset": SEED_OFFSET,
        "validation_seed_offset": VALIDATION_SEED_OFFSET,
        "predict_batch_size": PREDICT_BATCH_SIZE,
        "target_case_batch_size": TARGET_CASE_BATCH_SIZE,
        "target_processes": TARGET_PROCESSES,
    }


def create_validation_config() -> dict[str, dict[str, list[int]]]:
    return {}


def load_validation_dataset(generator, meta: dict, bitness: int) -> dict[str, np.ndarray]:
    ensure_validation_entry(generator, meta, bitness)
    validation = meta["validation"][str(bitness)]
    ids = [int(case_id) for case_id in validation["ids"]]
    x_validation = generate_sample_tensors(
        generator, bitness, ids, REPS, TARGET_PROCESSES
    )
    y_validation = np.asarray(
        [np.log1p(generator.case_nodes(bitness, case_id)) for case_id in ids],
        dtype=np.float32,
    )
    return {
        "x": x_validation,
        "y": y_validation,
    }


def ensure_validation_entry(generator, meta: dict, bitness: int) -> None:
    key = str(bitness)
    if key in meta["validation"]:
        return

    ids = generate_ids(
        generator,
        bitness,
        VALIDATION_SAMPLES,
        VALIDATION_SEED_OFFSET + bitness,
    )
    meta["validation"][key] = {
        "ids": ids,
    }
    save_experiment_meta(meta, META_PATH)
