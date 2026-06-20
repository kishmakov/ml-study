from __future__ import annotations

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
SAMPLE_MODES = ("random", "block")
SAMPLE_MODE_LABELS = {
    "random": "random",
    "block": "block",
}

META_PATH = DEFAULT_META_PATH


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

    states = {
        sample_mode: build_training_state(
            generator,
            train_ids,
            validation_ids,
            sample_mode,
        )
        for sample_mode in SAMPLE_MODES
    }

    for epoch in range(1, TRAIN_EPOCHS + 1):
        for sample_mode in SAMPLE_MODES:
            loader_progress = meta["progress"]["loaders"][sample_mode]
            if loader_progress["stage"] == "done" or loader_progress["epoch"] >= epoch:
                continue

            state = states[sample_mode]
            train_metrics = train_regression_epoch(
                state["model"],
                state["train_loader"],
                state["optimizer"],
                state["scheduler"],
                desc=f"depth {sample_mode} epoch {epoch}/{TRAIN_EPOCHS}",
            )
            validation_metrics = evaluate_regression_loader(
                state["model"],
                state["validation_loader"],
            )
            save_sample_mode_metrics(
                meta,
                config,
                state["model"],
                sample_mode,
                epoch,
                train_metrics,
                validation_metrics,
            )

    if meta["progress"]["stage"] != "done":
        meta["progress"]["stage"] = overall_stage(meta)
        save_experiment_meta(meta, META_PATH)


def build_config() -> dict[str, Any]:
    return {
        "model_dir": DEFAULT_MODEL_DIR,
        "meta_path": META_PATH,
        "checkpoint_paths": {
            sample_mode: checkpoint_path_for(sample_mode)
            for sample_mode in SAMPLE_MODES
        },
        "bitness": BITNESS,
        "train_samples": TRAIN_SAMPLES,
        "validation_samples": VALIDATION_SAMPLES,
        "train_epochs": TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "reps": REPS,
        "processes": PROCESSES,
        "seed": SEED,
        "sample_modes": list(SAMPLE_MODES),
    }


def load_or_create_depth_meta(generator, config: dict[str, Any]) -> dict[str, Any]:
    if exists(META_PATH):
        meta = load_experiment_meta(META_PATH)
        assert_resume_config(meta["config"], config)
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
            "global_step": 0,
            "loaders": initial_loader_progress(),
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


def build_plot_series() -> list[dict[str, Any]]:
    lines = []
    for sample_mode in SAMPLE_MODES:
        label = SAMPLE_MODE_LABELS[sample_mode]
        lines.extend(
            [
                {
                    "label": f"{label} train",
                    "where": {"loader": sample_mode},
                    "x_key": "epoch",
                    "mae_key": "train_mae",
                    "rmse_key": "train_rmse",
                },
                {
                    "label": f"{label} validation",
                    "where": {"loader": sample_mode},
                    "x_key": "epoch",
                    "mae_key": "validation_mae",
                    "rmse_key": "validation_rmse",
                },
            ]
        )
    return [
        {
            "name": "depth_metrics",
            "title": "Depth prediction metrics",
            "x_label": "Epoch",
            "lines": lines,
        }
    ]


def build_loader(
    generator,
    case_ids: list[int],
    *,
    sample_mode: str,
    shuffle: bool,
):
    return make_depth_sample_loader(
        generator,
        BITNESS,
        case_ids,
        REPS,
        BATCH_SIZE,
        PROCESSES,
        sample_mode=sample_mode,
        shuffle=shuffle,
        drop_last=shuffle,
    )


def build_training_state(
    generator,
    train_ids: list[int],
    validation_ids: list[int],
    sample_mode: str,
) -> dict[str, Any]:
    train_loader = build_loader(
        generator,
        train_ids,
        sample_mode=sample_mode,
        shuffle=True,
    )
    validation_loader = build_loader(
        generator,
        validation_ids,
        sample_mode=sample_mode,
        shuffle=False,
    )

    model = DeepSetPredictor(sample_point_dim(BITNESS))
    checkpoint_path = checkpoint_path_for(sample_mode)
    if exists(checkpoint_path):
        checkpoint = load_training_checkpoint(checkpoint_path, DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=5,
        factor=0.5,
    )
    return {
        "model": model,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def save_sample_mode_metrics(
    meta: dict[str, Any],
    config: dict[str, Any],
    model: nn.Module,
    sample_mode: str,
    epoch: int,
    train_metrics: dict[str, float],
    validation_metrics: dict[str, float],
) -> None:
    global_step = int(meta["progress"]["global_step"]) + 1
    progress = {
        "stage": "train" if epoch < TRAIN_EPOCHS else "done",
        "epoch": epoch,
    }
    metric = {
        "global_step": global_step,
        "epoch": epoch,
        "bitness": BITNESS,
        "loader": sample_mode,
        "train_loss": train_metrics["loss"],
        "train_rmse": train_metrics["rmse"],
        "train_mae": train_metrics["mad"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mad"],
    }
    meta.setdefault("metrics", []).append(metric)
    meta["progress"]["global_step"] = global_step
    meta["progress"]["loaders"][sample_mode] = progress
    meta["progress"]["stage"] = overall_stage(meta)
    save_training_checkpoint(model, config, progress, checkpoint_stem_for(sample_mode))
    save_experiment_meta(meta, META_PATH)
    print(
        f"depth loader={sample_mode} epoch={epoch:>3}  "
        f"train_rmse={metric['train_rmse']:.4f}  "
        f"train_mae={metric['train_mae']:.4f}  "
        f"validation_rmse={metric['validation_rmse']:.4f}  "
        f"validation_mae={metric['validation_mae']:.4f}"
    )


def initial_loader_progress() -> dict[str, dict[str, int | str]]:
    return {
        sample_mode: {
            "stage": "train",
            "epoch": 0,
        }
        for sample_mode in SAMPLE_MODES
    }


def overall_stage(meta: dict[str, Any]) -> str:
    if all(
        progress["stage"] == "done"
        for progress in meta["progress"]["loaders"].values()
    ):
        return "done"
    return "train"


def checkpoint_stem_for(sample_mode: str) -> str:
    return join(DEFAULT_MODEL_DIR, f"depth_{sample_mode}_b{BITNESS:02d}")


def checkpoint_path_for(sample_mode: str) -> str:
    return checkpoint_stem_for(sample_mode) + ".pt"


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


def train_regression_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    *,
    desc: str = "depth",
) -> dict[str, float]:
    criterion = nn.MSELoss()

    model.train()
    epoch_start = time.perf_counter()
    squared_error_sum = 0.0
    absolute_error_sum = 0.0
    count = 0

    for xb, yb in tqdm(loader, total=len(loader), desc=desc):
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
    print(
        f"{desc}  "
        f"loss={final_loss:.4f}  "
        f"elapsed={epoch_elapsed:.2f}s  "
        f"device={DEVICE}"
    )
    return {
        "loss": final_loss,
        "rmse": float(np.sqrt(final_loss)),
        "mad": float(absolute_error_sum / count),
    }


def assert_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    for key, value in current.items():
        assert saved[key] == value, (
            f"Cannot resume depth training with changed {key}: "
            f"saved={saved[key]!r}, current={value!r}"
        )
