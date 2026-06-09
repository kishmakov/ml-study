# Project

This project is a setup to test DeepCubeA algorithm for different puzzles.

# Implementation Details

- Keep assertion checks simple `assert foo, bar`, don't use ifs
- Avoid default values other than in test code
- Keep everything except test deterministic by explicitly asking for a seed
- Do not add package-presence guards (e.g. `assert torch is not None`); modules
  are imported directly and may be assumed available.

# Code Layout

- `puzzle/puzzle.py` — the `Puzzle` protocol. `scramble` is a concrete default method
  (reset to a random solved state via `solved_states()`, then a non-backtracking
  random walk), so puzzles inherit it for free. All scrambling takes an explicit
  `random.Random`.
- `puzzle/cube3.py`, `puzzle/n_puzzle.py` — concrete puzzles that subclass `Puzzle` to inherit
  `scramble`.
- `puzzle_factory.py` — name→`Puzzle` resolution and model/meta path helpers.
- `costtogo.py` — the value network (`CostToGoNet`) and heuristics
  (`CostToGo`, `NeuralCostToGo`). `NeuralCostToGo.from_checkpoint` is the single
  entry point for loading a trained model.
- `search/a_star.py` — batched weighted A* (`a_star_search`).
- `state_io.py` — all training I/O in one place: reading/writing checkpoints and
  run metadata, and the training log messages. The training loop itself stays
  free of disk and `print` calls.
- `train_costtogo.py` — the training loop and CTG evaluation.
- `plot_ctg.py` — plots metrics from `meta_*.json`.
- `daemon/client.py` — starts and coordinates local CTG worker processes over
  ZeroMQ.
- `daemon/worker.cpp` and `puzzle/environment.{hpp,cpp}` — native C++ worker
  and puzzle/input-vector logic used by the daemon.
- `build_native.sh` — builds the native worker binary at `daemon/worker`.

# Training

Python entry point:

```bash
uv run train_costtogo.py
```

Build the native daemon worker before running training that uses CTG
evaluation:

```bash
./build_native.sh
```

For quick Cube3 smoke tests, bound CTG search explicitly:

```bash
uv run train_costtogo.py --puzzle Cube3 --close-states 4 --bellman-states 2 --bellman-iterations 0 --ctg-count 1 --ctg-max-states 25
```

# Native Worker

- There is no Mojo implementation in this project.
- Training is Python/PyTorch.
- The native component is a C++ ZeroMQ worker used by `daemon/client.py`.
- The native worker currently deserializes Cube3 and NPuzzle states, generates
  the same model input vectors as the Python puzzle implementations, appends
  those vectors to `/tmp/cube/queries.txt`, and returns a dummy unsolved result.

# Search Optimization

CTG metric evaluation in Python uses a batched weighted A* variant inspired by
DeepCubeA's `parallel_weighted_astar.cpp`: it pops multiple frontier nodes,
expands their children, evaluates child heuristics as a batch, and then pushes
the scored children back into the open heap.
