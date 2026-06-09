"""Sliding N-puzzle description.

The state is a flat row-major permutation of ``0..dim**2 - 1`` where ``0`` is
the blank tile.  Actions follow the DeepCubeA N-puzzle environment names:
``U, D, L, R``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puzzle import Puzzle, StateFloat, StateKey


@dataclass(frozen=True, slots=True)
class NPuzzleState:
    tiles: tuple[int, ...]


_ACTION_NAMES = ("U", "D", "L", "R")
_INVERSE_ACTIONS = (1, 0, 3, 2)


def _solved_tiles(dim: int) -> tuple[int, ...]:
    return tuple(range(1, dim * dim)) + (0,)


def _state_to_key(state: NPuzzleState) -> StateKey:
    return ",".join(str(tile) for tile in state.tiles)


def _state_from_key(state_key: StateKey, dim: int) -> NPuzzleState:
    tiles = tuple(int(tile) for tile in state_key.split(","))
    state = NPuzzleState(tiles)
    _validate_state(state, dim)
    return state


def _validate_state(state: NPuzzleState, dim: int) -> None:
    assert dim >= 2, "NPuzzle dimension must be at least 2"
    assert len(state.tiles) == dim * dim, "NPuzzle state has wrong tile count"
    assert sorted(state.tiles) == list(range(dim * dim)), (
        "NPuzzle state is not a permutation"
    )


def _zero_position(state: NPuzzleState, dim: int) -> tuple[int, int, int]:
    zero_idx = state.tiles.index(0)
    row, col = divmod(zero_idx, dim)
    return zero_idx, row, col


def _swap_zero_idx(dim: int, zero_idx: int, action: int) -> int:
    row, col = divmod(zero_idx, dim)
    if action == 0:
        assert row < dim - 1, "Cannot apply U at bottom row"
        return zero_idx + dim
    if action == 1:
        assert row > 0, "Cannot apply D at top row"
        return zero_idx - dim
    if action == 2:
        assert col < dim - 1, "Cannot apply L at right edge"
        return zero_idx + 1
    assert action == 3, f"Invalid NPuzzle action {action!r}"
    assert col > 0, "Cannot apply R at left edge"
    return zero_idx - 1


class NPuzzle(Puzzle):
    """Stateful sliding puzzle with fixed inverse action ids."""

    action_names: tuple[str, ...] = _ACTION_NAMES
    inverse_actions: tuple[int, ...] = _INVERSE_ACTIONS

    def __init__(self, dim: int, state_key: StateKey | None = None) -> None:
        assert dim >= 2, "NPuzzle dimension must be at least 2"
        self.dim = dim
        self._solved_state = NPuzzleState(_solved_tiles(dim))
        self._state = (
            _state_from_key(state_key, dim) if state_key else self._solved_state
        )

    def solved_states(self) -> list[StateKey]:
        return [_state_to_key(self._solved_state)]

    @staticmethod
    def inverse_action(action: int) -> int:
        return NPuzzle.inverse_actions[NPuzzle._validate_action(action)]

    def state_key(self) -> StateKey:
        return _state_to_key(self._state)

    def reset(self, state_key: StateKey) -> None:
        self._state = _state_from_key(state_key, self.dim)

    def cost_to_go_input(self) -> StateFloat:
        return np.asarray(self._state.tiles, dtype=np.float32)

    def actions(self) -> tuple[int, ...]:
        _, row, col = _zero_position(self._state, self.dim)
        actions: list[int] = []
        if row < self.dim - 1:
            actions.append(0)
        if row > 0:
            actions.append(1)
        if col < self.dim - 1:
            actions.append(2)
        if col > 0:
            actions.append(3)
        return tuple(actions)

    def apply(self, action: int) -> tuple[StateKey, float]:
        action = self._validate_action(action)
        zero_idx, _, _ = _zero_position(self._state, self.dim)
        swap_idx = _swap_zero_idx(self.dim, zero_idx, action)

        tiles = list(self._state.tiles)
        tiles[zero_idx], tiles[swap_idx] = tiles[swap_idx], tiles[zero_idx]
        self._state = NPuzzleState(tuple(tiles))
        return self.state_key(), 1.0

    def is_solved(self) -> bool:
        return self._state == self._solved_state

    @staticmethod
    def _validate_action(action: int) -> int:
        assert 0 <= action < len(_ACTION_NAMES), f"Invalid NPuzzle action {action!r}"
        return action
