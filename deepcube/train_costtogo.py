"""Train Cube3 neural cost-to-go model.

The training loop mirrors the baseline DAVI shape: first fit exact near-goal
targets collected by BFS, then refine on Bellman targets from random walks.
Weights are written to ``/tmp/cube/model.pt``.
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import deque
from copy import deepcopy
from json import dump
from os import makedirs
from random import Random
from time import time
from typing import Any

import numpy as np

from costtogo import (
    CostToGoNet,
    META_PATH,
    MODEL_DIR,
    MODEL_PATH,
    count_params,
    nn,
    torch,
)
from cube3 import Cube3
from puzzle import StateKey


NUM_CLOSE_STATES = 4096
NUM_WALKS = 512
MAX_WALK = 30
PRE_TRAIN_EPOCHS = 4
BELLMAN_UPDATES = 4
TARGET_EPOCHS = 2
BATCH_SIZE = 256
EVAL_BATCH_SIZE = 1024
LR = 1e-3

TRAIN_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def collect_close_states(num_states: int) -> tuple[list[StateKey], np.ndarray]:
    cube = Cube3()
    solved = cube.solved_states()[0]
    queue = deque([(solved, 0)])
    seen = {solved}
    states = [solved]
    targets = [0.0]

    while queue and len(states) < num_states:
        state, distance = queue.popleft()
        for action in cube.actions():
            if len(states) >= num_states:
                break
            cube.reset(state)
            child, _ = cube.apply(action)
            if child in seen:
                continue
            seen.add(child)
            queue.append((child, distance + 1))
            states.append(child)
            targets.append(float(distance + 1))

    return states, np.asarray(targets, dtype=np.float32)


def collect_walk_states(num_walks: int, max_walk: int, seed: int) -> list[StateKey]:
    assert num_walks > 0, "num_walks must be positive"
    assert max_walk > 0, "max_walk must be positive"

    rng = Random(seed)
    states: list[StateKey] = []

    for _ in range(num_walks):
        cube = Cube3()
        previous_action: int | None = None
        walk_len = rng.randint(1, max_walk)
        for _depth in range(walk_len):
            candidates = list(cube.actions())
            if previous_action is not None:
                candidates.remove(cube.inverse_action(previous_action))
            action = rng.choice(candidates)
            cube.apply(action)
            states.append(cube.state_key())
            previous_action = action

    return list(dict.fromkeys(states))


def train_cube3_cost_to_go(
    seed: int,
    num_close_states: int,
    num_walks: int,
    max_walk: int,
    pre_train_epochs: int,
    bellman_updates: int,
    target_epochs: int,
    batch_size: int,
    eval_batch_size: int,
    lr: float,
    device: str,
    model_path: str,
    meta_path: str,
) -> dict[str, Any]:
    assert seed >= 0, "seed must be non-negative"

    started_at = time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    cube = Cube3()
    input_size = len(cube.cost_to_go_input())
    model = CostToGoNet.from_puzzle(cube).to(device)

    close_states, close_targets = collect_close_states(num_close_states)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history: list[dict[str, float | int | str]] = []

    for epoch in range(pre_train_epochs):
        loss = train_one_epoch(
            model,
            cube,
            close_states,
            close_targets,
            batch_size,
            opt,
            rng,
            device,
        )
        history.append({"stage": "pre", "epoch": epoch, "loss": loss})
        print(f"pre epoch={epoch} loss={loss:.4f}")

    target_model = clone_model(model)
    for update in range(bellman_updates):
        states = collect_walk_states(num_walks, max_walk, seed + update + 1)
        bellman = bellman_targets(cube, target_model, states, eval_batch_size, device)
        train_states = close_states + states
        train_targets = np.concatenate([close_targets, bellman])

        for epoch in range(target_epochs):
            loss = train_one_epoch(
                model,
                cube,
                train_states,
                train_targets,
                batch_size,
                opt,
                rng,
                device,
            )
            history.append({"stage": "bellman", "epoch": epoch, "loss": loss})
            print(f"update={update} epoch={epoch} loss={loss:.4f}")

        target_model = clone_model(model)

    makedirs(MODEL_DIR, exist_ok=True)
    config = {
        "num_close_states": num_close_states,
        "num_walks": num_walks,
        "max_walk": max_walk,
        "pre_train_epochs": pre_train_epochs,
        "bellman_updates": bellman_updates,
        "target_epochs": target_epochs,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "lr": lr,
        "seed": seed,
        "input_size": input_size,
        "params": count_params(model),
        "history": history,
    }
    torch.save({"state_dict": model.state_dict(), "config": config}, model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        dump(
            {
                "model_path": model_path,
                "wall_time_sec": time() - started_at,
                "config": config,
            },
            f,
            indent=2,
        )

    return config


def train_one_epoch(
    model: CostToGoNet,
    cube: Cube3,
    states: list[StateKey],
    targets: np.ndarray,
    batch_size: int,
    opt: Any,
    rng: np.random.Generator,
    device: str,
) -> float:
    assert torch is not None, "PyTorch is required for train_one_epoch"
    assert nn is not None, "PyTorch is required for train_one_epoch"
    assert len(states) == len(targets), "states and targets must have same length"

    model.train()
    loss_fn = nn.MSELoss()
    losses: list[float] = []
    order = rng.permutation(len(states))

    for start in range(0, len(states), batch_size):
        indices = order[start : start + batch_size]
        if len(indices) == 0:
            continue
        batch_states = [states[int(idx)] for idx in indices]
        x = states_to_tensor(cube, batch_states, device)
        y = torch.from_numpy(targets[indices].astype(np.float32)).to(device)

        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses))


def bellman_targets(
    cube: Cube3,
    target_model: CostToGoNet,
    states: list[StateKey],
    eval_batch_size: int,
    device: str,
) -> np.ndarray:
    targets = np.zeros(len(states), dtype=np.float32)
    next_states: list[StateKey] = []
    groups: list[tuple[int, int]] = []

    for state in states:
        cube.reset(state)
        if cube.is_solved():
            groups.append((len(next_states), len(next_states)))
            continue

        group_start = len(next_states)
        for action in cube.actions():
            cube.reset(state)
            child, _ = cube.apply(action)
            next_states.append(child)
        groups.append((group_start, len(next_states)))

    next_values = predict_values(cube, target_model, next_states, eval_batch_size, device)
    for idx, (group_start, group_end) in enumerate(groups):
        if group_start == group_end:
            targets[idx] = 0.0
        else:
            targets[idx] = 1.0 + float(np.min(next_values[group_start:group_end]))

    return targets


def predict_values(
    cube: Cube3,
    model: CostToGoNet,
    states: list[StateKey],
    batch_size: int,
    device: str,
) -> np.ndarray:
    assert torch is not None, "PyTorch is required for predict_values"

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            batch = states[start : start + batch_size]
            x = states_to_tensor(cube, batch, device)
            pred = model(x).cpu().numpy()
            preds.append(np.maximum(pred, 0.0).astype(np.float32))

    return np.concatenate(preds) if preds else np.asarray([], dtype=np.float32)


def states_to_tensor(cube: Cube3, states: list[StateKey], device: str) -> Any:
    assert torch is not None, "PyTorch is required for states_to_tensor"

    encoded = []
    for state in states:
        cube.reset(state)
        encoded.append(cube.cost_to_go_input())
    return torch.from_numpy(np.stack(encoded).astype(np.float32)).to(device)


def clone_model(model: CostToGoNet) -> CostToGoNet:
    target_model = deepcopy(model)
    target_model.eval()
    for parameter in target_model.parameters():
        parameter.requires_grad_(False)
    return target_model


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=239)
    args = parser.parse_args()

    train_cube3_cost_to_go(
        args.seed,
        NUM_CLOSE_STATES,
        NUM_WALKS,
        MAX_WALK,
        PRE_TRAIN_EPOCHS,
        BELLMAN_UPDATES,
        TARGET_EPOCHS,
        BATCH_SIZE,
        EVAL_BATCH_SIZE,
        LR,
        TRAIN_DEVICE,
        MODEL_PATH,
        META_PATH,
    )


if __name__ == "__main__":
    main()
