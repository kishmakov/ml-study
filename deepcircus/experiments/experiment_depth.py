from __future__ import annotations

from collections.abc import Callable
from os.path import exists, join
from typing import Any
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from experiments.generator import sample_point_dim
from experiments.model import DEVICE, DeepSetPredictor, regression_metrics
from experiments.sampler import generate_ids, make_depth_sample_loader
from experiments.state_io import (
    DEFAULT_META_PATH,
    DEFAULT_MODEL_DIR,
    ensure_model_dir,
    load_experiment_meta,
    load_training_checkpoint,
    save_experiment_meta,
    save_training_checkpoint,
)


BITNESS = 16
TRAIN_SAMPLES = 1 << 15
VALIDATION_SAMPLES = 1024

TRAIN_EPOCHS = 500
BATCH_SIZE = 256
LR = 1e-3

REPS = 128
PROCESSES = 16
SEED = 239

META_PATH = DEFAULT_META_PATH
CHECKPOINT_STEM = join(DEFAULT_MODEL_DIR, f"depth_b{BITNESS:02d}")
CHECKPOINT_PATH = CHECKPOINT_STEM + ".pt"


def run_experiment(generator) -> None:
    ensure_model_dir(DEFAULT_MODEL_DIR)
    config = build_config()
    meta = load_or_create_depth_meta(generator, config)
    progress = meta["progress"]
    if progress["stage"] == "done":
        print(f"training already complete: {META_PATH}")
        return

    train_ids = [int(case_id) for case_id in meta["ids"]["train"]]
    validation_ids = [int(case_id) for case_id in meta["ids"]["validation"]]

    train_loader = build_loader(generator, train_ids, shuffle=True)
    validation_loader = build_loader(generator, validation_ids, shuffle=False)

    model = DeepSetPredictor(sample_point_dim(BITNESS))
    if exists(CHECKPOINT_PATH):
        checkpoint = load_training_checkpoint(CHECKPOINT_PATH, DEVICE)
        model.load_state_dict(checkpoint["state_dict"])

    start_epoch = int(meta["progress"]["epoch"])

    def save_epoch_metrics(
        epoch: int,
        train_metrics: dict[str, float],
        _elapsed: float,
    ) -> None:
        validation_metrics = evaluate_regression_loader(model, validation_loader)
        global_step = int(meta["progress"]["global_step"]) + 1
        progress = {
            "stage": "train" if epoch < TRAIN_EPOCHS else "done",
            "epoch": epoch,
            "global_step": global_step,
        }
        metric = {
            "global_step": global_step,
            "epoch": epoch,
            "bitness": BITNESS,
            "train_loss": train_metrics["loss"],
            "train_rmse": train_metrics["rmse"],
            "train_mae": train_metrics["mad"],
            "validation_rmse": validation_metrics["rmse"],
            "validation_mae": validation_metrics["mad"],
        }
        meta.setdefault("metrics", []).append(metric)
        meta["progress"] = progress
        save_training_checkpoint(model, config, progress, CHECKPOINT_STEM)
        save_experiment_meta(meta, META_PATH)
        print(
            f"depth epoch={epoch:>3}  "
            f"train_rmse={metric['train_rmse']:.4f}  "
            f"train_mae={metric['train_mae']:.4f}  "
            f"validation_rmse={metric['validation_rmse']:.4f}  "
            f"validation_mae={metric['validation_mae']:.4f}"
        )

    train_regression_loader(
        model,
        train_loader,
        epochs=TRAIN_EPOCHS,
        lr=LR,
        start_epoch=start_epoch,
        on_epoch=save_epoch_metrics,
    )

    if meta["progress"]["stage"] != "done":
        meta["progress"] = {
            "stage": "done",
            "epoch": TRAIN_EPOCHS,
            "global_step": int(meta["progress"]["global_step"]),
        }
        save_experiment_meta(meta, META_PATH)


def build_config() -> dict[str, Any]:
    return {
        "model_dir": DEFAULT_MODEL_DIR,
        "meta_path": META_PATH,
        "checkpoint_path": CHECKPOINT_PATH,
        "bitness": BITNESS,
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "train_epochs": TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "reps": REPS,
        "processes": PROCESSES,
        "seed": SEED,
    }


def load_or_create_depth_meta(generator, config: dict[str, Any]) -> dict[str, Any]:
    if exists(META_PATH):
        meta = load_experiment_meta(META_PATH)
        assert_resume_config(meta["config"], config)
        ensure_depth_meta_shape(meta)
        print(f"resuming from {META_PATH}: progress={meta['progress']}")
        return meta

    ids = generate_ids(
        generator,
        BITNESS,
        TRAIN_SAMPLES + VALIDATION_SAMPLES,
        SEED,
    )
    meta = {
        "config": config,
        "progress": {
            "stage": "train",
            "epoch": 0,
            "global_step": 0,
        },
        "ids": {
            "train": ids[:TRAIN_SAMPLES],
            "validation": ids[TRAIN_SAMPLES:],
        },
        "metrics": [],
        "series": build_plot_series(),
    }
    save_experiment_meta(meta, META_PATH)
    return meta


def ensure_depth_meta_shape(meta: dict[str, Any]) -> None:
    changed = False
    for metric in meta.get("metrics", []):
        if "train_mad" in metric and "train_mae" not in metric:
            metric["train_mae"] = metric["train_mad"]
            changed = True
        if "validation_mad" in metric and "validation_mae" not in metric:
            metric["validation_mae"] = metric["validation_mad"]
            changed = True

    plot_series = build_plot_series()
    if meta.get("series") != plot_series:
        meta["series"] = plot_series
        meta.pop("plot_series", None)
        changed = True

    if changed:
        save_experiment_meta(meta, META_PATH)


def build_plot_series() -> list[dict[str, Any]]:
    return [
        {
            "name": "depth_metrics",
            "title": "Depth prediction metrics",
            "x_label": "Epoch",
            "lines": [
                {
                    "label": "train",
                    "where": {},
                    "x_key": "epoch",
                    "mae_key": "train_mae",
                    "rmse_key": "train_rmse",
                },
                {
                    "label": "validation",
                    "where": {},
                    "x_key": "epoch",
                    "mae_key": "validation_mae",
                    "rmse_key": "validation_rmse",
                },
            ],
        }
    ]


def build_loader(generator, case_ids: list[int], *, shuffle: bool):
    return make_depth_sample_loader(
        generator,
        BITNESS,
        case_ids,
        REPS,
        BATCH_SIZE,
        PROCESSES,
        shuffle=shuffle,
        drop_last=shuffle,
    )


def evaluate_regression_loader(
    model: nn.Module,
    loader: DataLoader,
) -> dict[str, float]:
    model.to(DEVICE)
    model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.float32, device=DEVICE).reshape(-1)
            predictions.append(model(xb).detach().cpu().numpy().ravel())
            targets.append(yb.cpu().numpy().ravel())

    return regression_metrics(
        np.concatenate(predictions).astype(np.float32),
        np.concatenate(targets).astype(np.float32),
    )


def train_regression_loader(
    model: nn.Module,
    loader: DataLoader,
    *,
    epochs: int,
    lr: float,
    start_epoch: int = 0,
    on_epoch: Callable[[int, dict[str, float], float], None] | None = None,
) -> float:
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
    )

    final_loss = float("inf")
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        epoch_start = time.perf_counter()
        squared_error_sum = 0.0
        absolute_error_sum = 0.0
        count = 0

        for xb, yb in tqdm(
            loader,
            total=len(loader),
            desc=f"depth epoch {epoch}/{epochs}",
        ):
            xb = torch.as_tensor(xb, dtype=torch.float32, device=DEVICE)
            yb = torch.as_tensor(yb, dtype=torch.float32, device=DEVICE).reshape(-1, 1)
            optimizer.zero_grad()
            prediction = model(xb)
            loss = criterion(prediction, yb)
            loss.backward()
            optimizer.step()

            errors = (prediction.detach() - yb).reshape(-1)
            squared_error_sum += torch.sum(errors.square()).item()
            absolute_error_sum += torch.sum(torch.abs(errors)).item()
            count += len(xb)

        final_loss = float(squared_error_sum / count)
        scheduler.step(final_loss)
        epoch_elapsed = time.perf_counter() - epoch_start
        train_metrics = {
            "loss": final_loss,
            "rmse": float(np.sqrt(final_loss)),
            "mad": float(absolute_error_sum / count),
        }

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch:>3}/{epochs}  "
                f"loss={final_loss:.4f}  "
                f"elapsed={epoch_elapsed:.2f}s  "
                f"device={DEVICE}"
            )

        if on_epoch is not None:
            on_epoch(epoch, train_metrics, epoch_elapsed)

    return final_loss


def assert_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for key, value in current.items():
        assert saved[key] == value, (
            f"Cannot resume depth training with changed {key}: "
            f"saved={saved[key]!r}, current={value!r}"
        )
