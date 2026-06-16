# Project

This is a research project to study ML approach to handle decision trees.

# Implementation Details

- Keep assertion checks simple `assert foo, bar`, don't use ifs
- In C++ use plain asserts `assert(condition)`
- Keep C++ generation deterministic from `(bitness, case_id)`
- Do not add package-presence guards (e.g. `assert torch is not None`)
- Do not generalize code for running
- The generator API is bitness-based: use `uint16_t bitness`, not series ids or bit masks
- `generator_case_value` receives exactly `bitness` input characters
- Bitness is always a parameter
- For bitness `0..4`, generate exact truth-table trees and cache them under `tmp/bN`
- For larger bitness, use the random tree generator over bits `0..bitness-1`

# Code Layout

- `tmp` is the directory not indexed by git
- `generator/decision_tree.{h,cpp}` owns `DecisionTree`, `Div`, `Node`, and tree evaluation/building
- `generator/small_bitness.{h,cpp}` owns exact small-bitness solving and cache read/write
- `generator/generator.{h,cpp}` owns the public C API and main generation dispatch
- `experiments/generator.py` owns generator loading, ctypes signatures, the Python generator wrapper, and sample generation helpers
- `experiments/experiment_*.py` should contain experiment logic only; do not put ctypes or shared-library details there
- `scripts/*.py` should stay thin entrypoints over experiment/generator helpers
- `scripts/inspect.py` inspects small-bitness trees via `experiments.generator`


# Building

- `scripts/build_generator.sh` builds generator in `build` directory;
   this is where generator is supposed to be stored

# Running

```bash
uv run scripts/run.py
```
