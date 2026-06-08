"""Helpers for selecting puzzle implementations from CLI names."""

from __future__ import annotations

from re import fullmatch

from cube3 import Cube3
from n_puzzle import NPuzzle
from puzzle import Puzzle
from costtogo import NeuralCostToGo


DEFAULT_PUZZLE = "Cube3"
MODEL_DIR = "/tmp/cube"
PUZZLE_HELP = "Puzzle name: Cube3, NPuzzle3, NPuzzle4"


def create_puzzle(puzzle_name: str) -> Puzzle:
    key = _normalize_puzzle_name(puzzle_name)

    if key == "cube3":
        return Cube3()

    dim_match = fullmatch(r"npuzzle(\d+)", key)
    if dim_match:
        dim = int(dim_match.group(1))
        assert dim >= 2, "NPuzzle dimension must be at least 2"
        return NPuzzle(dim)

    raise AssertionError(f"Unknown puzzle {puzzle_name!r}. {PUZZLE_HELP}.")


def _normalize_puzzle_name(puzzle_name: str) -> str:
    return (
        puzzle_name.strip()
        .lower()
        .replace("_", "")
        .replace("-", "")
        .replace(" ", "")
    )


def model_stem_for(name: str) -> str:
    return f"{MODEL_DIR}/model_{name.lower()}"


def load_cost_to_go(puzzle: Puzzle, checkpoint_path: str, device: str) -> NeuralCostToGo:
    return NeuralCostToGo.from_checkpoint(checkpoint_path, puzzle, device)

