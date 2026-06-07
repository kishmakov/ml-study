# Project

This project is a setup to test DeepCubeA algorithm for different puzzles.

# Implementation Details

- Keep assertion checks simple `assert foo, bar`, don't use ifs
- Avoid default values other than in test code
- Keep everything except test deterministic by explicitly asking for a seed
- Do not add package-presence guards (e.g. `assert torch is not None`); modules
  are imported directly and may be assumed available.
- The Mojo training path must not delegate to Python. Do not import
  `std.python` from training code.

# Code Layout

- `puzzle.py` — the `Puzzle` protocol. `scramble` is a concrete default method
  (reset to a random solved state via `solved_states()`, then a non-backtracking
  random walk), so puzzles inherit it for free. All scrambling takes an explicit
  `random.Random`.
- `cube3.py`, `n_puzzle.py` — concrete puzzles that subclass `Puzzle` to inherit
  `scramble`.
- `puzzle_factory.py` — name→`Puzzle` resolution and model/meta path helpers.
- `costtogo.py` — the value network (`CostToGoNet`) and heuristics
  (`CostToGo`, `NeuralCostToGo`). `NeuralCostToGo.from_checkpoint` is the single
  entry point for loading a trained model.
- `search_a_star.py` — batched weighted A* (`solve_a_star`).
- `state_io.py` — all training I/O in one place: reading/writing checkpoints and
  run metadata, and the training log messages. The training loop itself stays
  free of disk and `print` calls so it ports cleanly to Mojo.
- `train_costtogo.py` — the training loop and CTG evaluation.
- `plot_ctg.py` — plots metrics from `meta_*.json`.

# Training

Python entry point kept for reference while the rewrite is in progress:

```bash
uv run train_costtogo.py
```

Native Mojo entry point:

```bash
uv run mojo train_costtogo.mojo --puzzle Cube3 --close-states 4 --bellman-states 4 --bellman-iterations 1 --model-dir /tmp/cube
```

Build and run a compiled Mojo executable:

```bash
uv run mojo build train_costtogo.mojo -o /tmp/deepcube_train_costtogo
/tmp/deepcube_train_costtogo --puzzle Cube3 --close-states 4 --bellman-states 4 --bellman-iterations 1 --model-dir /tmp/cube
```

For quick Cube3 smoke tests, bound CTG search explicitly:

```bash
uv run mojo train_costtogo.mojo --puzzle Cube3 --close-states 4 --bellman-states 2 --bellman-iterations 0 --ctg-count 1 --ctg-max-states 25 --model-dir /tmp/cube
```

Current native Mojo status:

- `train_costtogo.mojo` is native Mojo and does not delegate to Python.
- The native path currently implements deterministic Cube3 and NPuzzle2/3/4
  training with the same DeepCube-style value-network shape as the Python
  reference (`512 -> 384 -> 2 residual blocks -> scalar`), BFS close-state
  pretraining with depth-based repetitions, Bellman fitting to a loss
  threshold, and CTG evaluation by weighted A*. Cube3 uses the same
  324-dimensional one-hot face-color input shape as the Python reference.
- The Mojo trainer writes native JSON artifacts under `--model-dir`:
  `model_*.json` contains value-network weights, and `meta_*.json` contains
  config, progress, and CTG metrics.
- The Python path still contains the fuller PyTorch experiment machinery. Full
  Mojo parity still requires PyTorch-compatible checkpoint loading/export,
  resumable checkpoint loading, and CTG plotting.

# Search Optimization

CTG metric evaluation in Python uses a batched weighted A* variant inspired by
DeepCubeA's `parallel_weighted_astar.cpp`: it pops multiple frontier nodes,
expands their children, evaluates child heuristics as a batch, and then pushes
the scored children back into the open heap.
