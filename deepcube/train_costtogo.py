"""Train a neural cost-to-go model.

The training loop mirrors the baseline DAVI shape: first fit exact near-goal
targets collected by BFS, then refine on Bellman targets from random walks.
Weights are written to ``/tmp/cube/model.pt`` by default.
"""

from __future__ import annotations

from argparse import ArgumentParser
from atexit import register
from collections import deque
from os.path import exists
from random import Random, seed as set_python_random_seed
from typing import Any

import numpy as np
import torch

from costtogo import CostToGoNet, count_params
from daemon.client import CtgZmqClient
from puzzle import Puzzle, StateKey
from puzzle_factory import (
    DEFAULT_PUZZLE,
    MODEL_DIR,
    PUZZLE_HELP,
    create_puzzle,
    model_stem_for
)
from state_io import (
    ensure_model_dir,
    load_training_checkpoint,
    load_training_meta,
    log_bellman_update,
    log_ctg_eval,
    log_pre_epoch,
    log_resume,
    log_training_start,
    log_until_fit_epoch,
    save_training_checkpoint,
    save_training_meta,
    save_training_state,
)


MAX_STEPS = 30
PRE_TRAIN_EPOCHS = 10
BATCH_SIZE = 512
EVAL_BATCH_SIZE = 1024
LR = 1e-3
TARGET_LOSS_THRESHOLD = 0.05
TARGET_MAX_EPOCHS = 16
CLOSE_STATE_BASE_REPETITIONS = 32

CTG_EVAL_DEPTHS = ( 5,  7, 10, 13, 15, 17, 20, 30)
CTG_EVAL_COUNT  = (20, 20, 20, 20, 20, 20, 20, 20)

# CTG_EVAL_DEPTHS = ( 5,  7, 10)
# CTG_EVAL_COUNT  = (20, 20, 20)

CTG_EVAL_MAX_STATES = 65_000
CTG_EVAL_WEIGHT = 0.1
CTG_EVAL_POP_BATCH_SIZE = 64

TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_CTG_EVAL_EXECUTOR: CtgZmqClient | None = None


def collect_close_states(
    num_states: int,
    puzzle: Puzzle,
) -> tuple[list[StateKey], np.ndarray]:
    solved_states = puzzle.solved_states()
    queue = deque([(solved, 0) for solved in solved_states])
    seen = {solved for solved in solved_states}
    states: list[StateKey] = []
    targets: list[float] = []

    for solved in solved_states:
        append_close_state_repetitions(states, targets, solved, 0, num_states)
        if len(states) >= num_states:
            break

    while queue and len(states) < num_states:
        state, distance = queue.popleft()
        puzzle.reset(state)
        for action in puzzle.actions():
            if len(states) >= num_states:
                break
            puzzle.reset(state)
            child, _ = puzzle.apply(action)
            if child in seen:
                continue
            seen.add(child)
            child_distance = distance + 1
            queue.append((child, child_distance))
            append_close_state_repetitions(
                states,
                targets,
                child,
                child_distance,
                num_states,
            )

    return states, np.asarray(targets, dtype=np.float32)


def append_close_state_repetitions(
    states: list[StateKey],
    targets: list[float],
    state: StateKey,
    distance: int,
    max_states: int,
) -> None:
    for _ in range(min(close_state_repetitions(distance), max_states - len(states))):
        states.append(state)
        targets.append(float(distance))


def close_state_repetitions(distance: int) -> int:
    return max(CLOSE_STATE_BASE_REPETITIONS >> distance, 1)


def collect_random_states(
    num_states: int,
    max_steps: int,
    rng: Random,
    puzzle: Puzzle,
) -> list[StateKey]:
    assert num_states > 0, "num_states must be positive"
    assert max_steps > 0, "max_walk must be positive"

    states: list[StateKey] = []

    for _ in range(num_states):
        steps = rng.randint(1, max_steps)
        states.append(puzzle.scramble(steps, rng)[0])

    return list(dict.fromkeys(states))


def train_puzzle_cost_to_go(
    puzzle_name: str,
    seed: int,
    num_close_states: int,
    num_bellman_states: int,
    max_walk: int,
    pre_train_epochs: int,
    bellman_updates: int,
    target_loss_threshold: float,
    target_max_epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
) -> dict[str, Any]:
    assert seed >= 0, "seed must be non-negative"

    ensure_model_dir(MODEL_DIR)

    torch.manual_seed(seed)

    puzzle = create_puzzle(puzzle_name)
    input_size = len(puzzle.cost_to_go_input())
    model = CostToGoNet.from_puzzle(puzzle).to(TRAIN_DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model_stem = model_stem_for(puzzle_name)
    history: list[dict[str, float | int | str]] = []
    progress = _initial_progress()

    log_training_start(
        puzzle_name,
        puzzle.cost_to_go_input().shape,
        count_params(model),
        TRAIN_DEVICE,
        num_close_states,
        num_bellman_states,
    )

    config = {
        "puzzle": puzzle_name,
        "num_close_states": num_close_states,
        "num_bellman_states": num_bellman_states,
        "max_walk": max_walk,
        "pre_train_epochs": pre_train_epochs,
        "bellman_updates": bellman_updates,
        "target_loss_threshold": target_loss_threshold,
        "target_max_epochs": target_max_epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "lr": lr,
        "seed": seed,
        "input_size": input_size,
        "params": count_params(model),
        "history": history,
    }

    meta_path = model_stem + ".json"
    if exists(meta_path):
        meta = load_training_meta(meta_path)
        assert "progress" in meta, f"invalid meta without progress: {meta_path}"
        assert "ctg_eval" in meta["config"], (
            f"invalid meta without ctg_eval: {meta_path}"
        )
        config["ctg_eval"] = meta["config"]["ctg_eval"]
        _assert_resume_config(meta["config"], config)
        checkpoint_path = model_stem + ".pt"
        assert exists(checkpoint_path), f"checkpoint not found at {checkpoint_path}"

        checkpoint = load_training_checkpoint(checkpoint_path, TRAIN_DEVICE)
        assert "state_dict" in checkpoint, f"invalid checkpoint: {checkpoint_path}"
        assert "optimizer_state_dict" in checkpoint, (
            f"invalid checkpoint without optimizer state: {checkpoint_path}"
        )
        model.load_state_dict(checkpoint["state_dict"])
        opt.load_state_dict(checkpoint["optimizer_state_dict"])
        history = meta["config"]["history"]
        config["history"] = history
        progress = meta["progress"]
        assert "seed_offset" in progress, (
            f"invalid progress without seed_offset: {meta_path}"
        )
        log_resume(meta_path, progress)
    else:
        config["ctg_eval"] = create_ctg_eval_config(seed, puzzle)

    close_states, close_targets = collect_close_states(num_close_states, puzzle)

    progress = run_pretraining_phase(
        model,
        opt,
        puzzle,
        close_states,
        close_targets,
        config,
        progress,
        seed,
        pre_train_epochs,
        batch_size,
        model_stem,
    )
    progress = run_bellman_phase(
        model,
        opt,
        puzzle,
        close_states,
        close_targets,
        config,
        progress,
        seed,
        num_bellman_states,
        max_walk,
        pre_train_epochs,
        bellman_updates,
        target_loss_threshold,
        target_max_epochs,
        batch_size,
        eval_batch_size,
        model_stem,
    )
    finalize_training_state(
        model,
        opt,
        config,
        progress,
        pre_train_epochs,
        bellman_updates,
        model_stem,
    )

    return config


def run_pretraining_phase(
    model: CostToGoNet,
    opt: Any,
    puzzle: Puzzle,
    close_states: list[StateKey],
    close_targets: np.ndarray,
    config: dict[str, Any],
    progress: dict[str, int | str],
    seed: int,
    pre_train_epochs: int,
    batch_size: int,
    model_stem: str,
) -> dict[str, int | str]:
    history = config["history"]
    trained = False

    for epoch in range(int(progress["pre_epoch"]), pre_train_epochs):
        trained = True
        seed_offset = int(progress["seed_offset"])
        _py_rng, np_rng = reset_epoch_randomness(seed + seed_offset)
        avg_loss = train_one_epoch(
            model,
            puzzle,
            close_states,
            close_targets,
            batch_size,
            opt,
            np_rng,
        )
        history.append(
            {"stage": "pre", "epoch": epoch, "loss": avg_loss, "seed_offset": seed_offset}
        )
        log_pre_epoch(epoch, avg_loss)

        progress = _pre_progress(epoch + 1, pre_train_epochs, seed_offset + 1)

    if trained:
        save_checkpoint_then_update_ctg_meta(
            model,
            opt,
            config,
            progress,
            model_stem,
        )

    return progress


def run_bellman_phase(
    model: CostToGoNet,
    opt: Any,
    puzzle: Puzzle,
    close_states: list[StateKey],
    close_targets: np.ndarray,
    config: dict[str, Any],
    progress: dict[str, int | str],
    seed: int,
    num_bellman_states: int,
    max_walk: int,
    pre_train_epochs: int,
    bellman_updates: int,
    target_loss_threshold: float,
    target_max_epochs: int,
    batch_size: int,
    eval_batch_size: int,
    model_stem: str,
) -> dict[str, int | str]:
    history = config["history"]
    bellman_start = int(progress["bellman_update"])

    for update in range(bellman_start, bellman_updates):
        seed_offset = int(progress["seed_offset"])
        py_rng, np_rng = reset_epoch_randomness(seed + seed_offset)
        b_states = collect_random_states(num_bellman_states, max_walk, py_rng, puzzle)
        b_targets = bellman_targets(
            puzzle,
            model,
            b_states,
            eval_batch_size,
        )

        topup = min(len(close_states), num_bellman_states - len(b_states))
        if topup > 0:
            idx = np_rng.choice(len(close_states), size=topup, replace=False)
            b_states.extend(close_states[i] for i in idx)
            b_targets = np.concatenate(
                [b_targets, close_targets[idx].astype(np.float32)]
            )

        train_result = train_until_fit(
            model,
            puzzle,
            b_states,
            b_targets,
            batch_size,
            opt,
            np_rng,
            target_loss_threshold,
            target_max_epochs,
        )
        history.append(
            {
                "stage": "bellman",
                "update": update,
                "loss": train_result["loss"],
                "epochs": train_result["epochs"],
                "seed_offset": seed_offset,
            }
        )
        log_bellman_update(update, train_result["loss"], train_result["epochs"])

        progress = _bellman_progress(
            pre_train_epochs,
            update + 1,
            bellman_updates,
            seed_offset + 1,
        )
        save_checkpoint_then_update_ctg_meta(
            model,
            opt,
            config,
            progress,
            model_stem,
        )

    return progress


def save_checkpoint_then_update_ctg_meta(
    model: CostToGoNet,
    opt: Any,
    config: dict[str, Any],
    progress: dict[str, int | str],
    model_stem: str,
) -> None:
    save_training_checkpoint(model, opt, config, progress, model_stem)
    append_ctg_metric(config, progress, model_stem)
    save_training_meta(config, progress, model_stem)


def finalize_training_state(
    model: CostToGoNet,
    opt: Any,
    config: dict[str, Any],
    progress: dict[str, int | str],
    pre_train_epochs: int,
    bellman_updates: int,
    model_stem: str,
) -> None:
    meta_path = model_stem + ".json"
    if exists(meta_path) and progress["stage"] == "done":
        return

    progress = _bellman_progress(
        pre_train_epochs,
        bellman_updates,
        bellman_updates,
        int(progress["seed_offset"]),
    )
    save_training_state(model, opt, config, progress, model_stem)


def create_ctg_eval_config(seed: int, puzzle: Puzzle) -> dict[str, Any]:
    rng = Random(seed)
    bunches = {
        depth: [puzzle.scramble(depth, rng)[0] for _ in range(reps)]
        for depth, reps in zip(CTG_EVAL_DEPTHS, CTG_EVAL_COUNT)
    }

    return {
        "bunches": bunches,
        "metrics": [],
    }


def append_ctg_metric(
    config: dict[str, Any],
    progress: dict[str, int | str],
    model_stem: str,
) -> None:
    ctg_eval = config["ctg_eval"]
    metric = {
        "iteration": len(ctg_eval["metrics"]) + 1,
        "progress": progress,
        "depths": {},
    }
    model_version = int(metric["iteration"])
    executor = get_ctg_eval_executor()
    values_by_depth, solved_by_depth = executor.evaluate(
        config["puzzle"],
        model_stem,
        model_version,
        ctg_eval["bunches"],
        "ctg eval",
    )

    for depth, states in ctg_eval["bunches"].items():
        depth = int(depth)
        values = values_by_depth.get(depth, [])
        solved = solved_by_depth.get(depth, [])
        assert len(values) == len(states), f"CTG evaluation lost depth={depth} values"
        assert len(solved) == len(states), f"CTG evaluation lost depth={depth} results"

        metric["depths"][str(depth)] = {
            "percent_solved": 100.0 * sum(solved) / len(solved),
            "avg_cost_to_go": float(np.mean(values)) if len(values) else 0.0,
        }

    ctg_eval["metrics"].append(metric)
    log_ctg_eval(metric["iteration"], ctg_eval["bunches"])


def get_ctg_eval_executor() -> CtgZmqClient:
    global _CTG_EVAL_EXECUTOR

    if _CTG_EVAL_EXECUTOR is None:
        _CTG_EVAL_EXECUTOR = CtgZmqClient.start()

    return _CTG_EVAL_EXECUTOR


def shutdown_ctg_eval_executor() -> None:
    global _CTG_EVAL_EXECUTOR

    if _CTG_EVAL_EXECUTOR is None:
        return

    _CTG_EVAL_EXECUTOR.close()
    _CTG_EVAL_EXECUTOR = None


register(shutdown_ctg_eval_executor)


def reset_epoch_randomness(epoch_seed: int) -> tuple[Random, np.random.Generator]:
    set_python_random_seed(epoch_seed)
    np.random.seed(epoch_seed)
    torch.manual_seed(epoch_seed)
    if TRAIN_DEVICE.startswith("cuda"):
        torch.cuda.manual_seed_all(epoch_seed)
    return Random(epoch_seed), np.random.default_rng(epoch_seed)


def _initial_progress() -> dict[str, int | str]:
    return {
        "stage": "pre",
        "pre_epoch": 0,
        "bellman_update": 0,
        "seed_offset": 0,
    }


def _pre_progress(
    pre_epoch: int,
    pre_train_epochs: int,
    seed_offset: int,
) -> dict[str, int | str]:
    stage = "bellman" if pre_epoch >= pre_train_epochs else "pre"
    return {
        "stage": stage,
        "pre_epoch": pre_epoch,
        "bellman_update": 0,
        "seed_offset": seed_offset,
    }


def _bellman_progress(
    pre_train_epochs: int,
    bellman_update: int,
    bellman_updates: int,
    seed_offset: int,
) -> dict[str, int | str]:
    stage = "done" if bellman_update >= bellman_updates else "bellman"
    return {
        "stage": stage,
        "pre_epoch": pre_train_epochs,
        "bellman_update": bellman_update,
        "seed_offset": seed_offset,
    }


def _assert_resume_config(saved: dict[str, Any], current: dict[str, Any]) -> None:
    keys = (
        "puzzle",
        "num_close_states",
        "num_bellman_states",
        "max_walk",
        "pre_train_epochs",
        "bellman_updates",
        "target_loss_threshold",
        "target_max_epochs",
        "batch_size",
        "eval_batch_size",
        "lr",
        "seed",
        "input_size",
    )
    for key in keys:
        assert saved[key] == current[key], (
            f"Cannot resume training with changed {key}: "
            f"saved={saved[key]!r}, current={current[key]!r}"
        )


def train_one_epoch(
    model: CostToGoNet,
    puzzle: Puzzle,
    states: list[StateKey],
    targets: np.ndarray,
    batch_size: int,
    opt: Any,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    assert len(states) == len(targets), "states and targets must have same length"

    model.train()
    loss_sum = 0.0
    num_examples = 0
    steps = 0
    order = rng.permutation(len(states))

    for start in range(0, len(states), batch_size):
        indices = order[start : start + batch_size]
        assert len(indices) > 0, "batch_size must be positive"
        batch_states = [states[int(idx)] for idx in indices]
        x = states_to_tensor(puzzle, batch_states)
        y = torch.from_numpy(targets[indices].astype(np.float32)).to(TRAIN_DEVICE)

        pred = model(x)
        per_example_loss = (pred - y).pow(2)
        loss = per_example_loss.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += per_example_loss.sum().item()
        num_examples += len(indices)
        steps += 1

    assert num_examples == len(states), "train_one_epoch must process all examples"
    return loss_sum / num_examples


def train_until_fit(
    model: CostToGoNet,
    puzzle: Puzzle,
    states: list[StateKey],
    targets: np.ndarray,
    batch_size: int,
    opt: Any,
    rng: np.random.Generator,
    loss_threshold: float,
    max_epochs: int,
) -> dict[str, float | int]:
    epoch = 0
    last_loss = float("inf")

    while epoch < max_epochs and last_loss > loss_threshold:
        last_loss = train_one_epoch(
            model,
            puzzle,
            states,
            targets,
            batch_size,
            opt,
            rng,
        )
        log_until_fit_epoch(epoch, last_loss)

        epoch += 1

    return {
        "loss": last_loss,
        "epochs": epoch,
    }


def bellman_targets(
    puzzle: Puzzle,
    model: CostToGoNet,
    states: list[StateKey],
    eval_batch_size: int,
) -> np.ndarray:
    targets = np.zeros(len(states), dtype=np.float32)
    next_states: list[StateKey] = []
    groups: list[tuple[int, int]] = []

    for state in states:
        puzzle.reset(state)
        if puzzle.is_solved():
            groups.append((len(next_states), len(next_states)))
            continue

        group_start = len(next_states)
        for action in puzzle.actions():
            puzzle.reset(state)
            child, _ = puzzle.apply(action)
            next_states.append(child)
        groups.append((group_start, len(next_states)))

    next_values = predict_values(puzzle, model, next_states, eval_batch_size)
    for idx, (group_start, group_end) in enumerate(groups):
        if group_start == group_end:
            targets[idx] = 0.0
        else:
            targets[idx] = 1.0 + float(np.min(next_values[group_start:group_end]))

    return targets


def predict_values(
    puzzle: Puzzle,
    model: CostToGoNet,
    states: list[StateKey],
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            x = states_to_tensor(puzzle, batch)
            pred = model(x).cpu().numpy()
            preds.append(np.maximum(pred, 0.0).astype(np.float32))

    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


def states_to_tensor(puzzle: Puzzle, states: list[StateKey]) -> Any:
    encoded = []
    for state in states:
        puzzle.reset(state)
        encoded.append(puzzle.cost_to_go_input())
    return torch.from_numpy(np.stack(encoded).astype(np.float32)).to(TRAIN_DEVICE)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--puzzle", default=DEFAULT_PUZZLE, help=PUZZLE_HELP)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--close-states", type=int, default=32*1024)
    parser.add_argument("--bellman-states", type=int, default=32*1024)
    parser.add_argument("--bellman-iterations", type=int, default=1024*1024)
    args = parser.parse_args()

    train_puzzle_cost_to_go(
        args.puzzle,
        args.seed,
        args.close_states,
        args.bellman_states,
        MAX_STEPS,
        PRE_TRAIN_EPOCHS,
        args.bellman_iterations,
        TARGET_LOSS_THRESHOLD,
        TARGET_MAX_EPOCHS,
        BATCH_SIZE,
        EVAL_BATCH_SIZE,
        LR,
    )


if __name__ == "__main__":
    main()
