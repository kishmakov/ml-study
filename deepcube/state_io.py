"""Disk persistence and log messages for cost-to-go training.

Centralizes the two process boundaries of the training loop: reading and
writing training state (model checkpoints and run metadata) and emitting the
human-readable progress lines. Keeping all of this in one module leaves the
training loop itself free of I/O.
"""

from __future__ import annotations

from collections.abc import Iterable
from json import dump, load
from os import makedirs
from os.path import splitext
from time import time
from typing import Any

import torch


# ---------------------------------------------------------------------------
# Training-state read / write
# ---------------------------------------------------------------------------

def ensure_model_dir(model_dir: str) -> None:
    makedirs(model_dir, exist_ok=True)


def load_training_meta(meta_path: str) -> dict[str, Any]:
    with open(meta_path, encoding="utf-8") as f:
        return load(f)


def load_training_checkpoint(model_path: str, device: str) -> dict[str, Any]:
    return torch.load(model_path, map_location=device, weights_only=False)


def save_training_checkpoint(
    model: Any,
    opt: Any,
    config: dict[str, Any],
    progress: dict[str, int | str],
    model_stem: str,
) -> None:
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "config": config,
        "progress": progress,
    }
    was_training = model.training
    model.eval()
    try:
        torch.save(checkpoint, model_stem + ".pt")
        scripted = torch.jit.script(model)
        scripted.save(model_stem + ".torchscript")
    finally:
        model.train(was_training)


def save_training_meta(
    config: dict[str, Any],
    progress: dict[str, int | str],
    model_stem: str,
    started_at: float,
) -> None:
    with open(model_stem + ".json", "w", encoding="utf-8") as f:
        dump(
            {
                "wall_time_sec": time() - started_at,
                "progress": progress,
                "config": config,
            },
            f,
            indent=2,
        )


def save_training_state(
    model: Any,
    opt: Any,
    config: dict[str, Any],
    progress: dict[str, int | str],
    model_stem: str,
    started_at: float,
) -> None:
    save_training_checkpoint(model, opt, config, progress, model_stem)
    save_training_meta(config, progress, model_stem, started_at)


# ---------------------------------------------------------------------------
# Training log messages
# ---------------------------------------------------------------------------

def log_training_start(
    puzzle_name: str,
    input_shape: Any,
    params: int,
    device: str,
    num_close_states: int,
    num_bellman_states: int,
) -> None:
    print(
        f"training puzzle={puzzle_name} input={input_shape}",
        f"params={params} device={device}",
        f"num_close_states={num_close_states}",
        f"num_bellman_states={num_bellman_states}",
    )


def log_resume(meta_path: str, progress: dict[str, int | str]) -> None:
    print(f"resuming from {meta_path}: progress={progress}")


def log_pre_epoch(epoch: int, loss: float) -> None:
    print(f"pre epoch={epoch} loss={loss:.4f}")


def log_bellman_update(update: int, loss: float, epochs: int) -> None:
    print(f"update={update} loss={loss:.4f} last={loss:.4f} epochs={epochs}")


def log_until_fit_epoch(epoch: int, loss: float) -> None:
    print(f"  until_fit epoch={epoch} loss={loss:.4f}")


def log_ctg_eval(iteration: int, depths: Iterable[Any]) -> None:
    print(
        f"ctg eval iteration={iteration} "
        f"depths={','.join(str(depth) for depth in depths)}"
    )
