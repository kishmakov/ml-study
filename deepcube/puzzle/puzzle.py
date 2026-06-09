"""Generic puzzle interfaces used by search algorithms."""

from __future__ import annotations

from collections.abc import Iterable
from random import Random
from typing import Protocol, TypeAlias

import numpy as np


StateKey: TypeAlias = str
StateFloat: TypeAlias = np.ndarray


class Puzzle(Protocol):
    """Stateful puzzle interface.

    Implementations keep their current state internally. Search code can use
    ``state_key()`` for bookkeeping and ``cost_to_go_input()`` for heuristics.
    It can call ``reset(state_key)`` to restore arbitrary states.
    """

    action_names: tuple[str, ...]
    """Action names indexed by integer action id."""

    def solved_states(self) -> list[StateKey]:
        """Return serialized states that should be treated as solved."""

    def inverse_action(self, action: int) -> int:
        """Return the inverse to given action."""

    def state_key(self) -> StateKey:
        """Return the serialized current state."""

    def reset(self, state_key: StateKey) -> None:
        """Set the current state to ``state_key``."""

    def cost_to_go_input(self) -> StateFloat:
        """Return the current state representation used by cost-to-go."""

    def actions(self) -> Iterable[int]:
        """Return integer action ids available from the current state."""

    def apply(self, action: int) -> tuple[StateKey, float]:
        """Apply ``action`` and return ``(state_key, cost)``."""

    def is_solved(self) -> bool:
        """Return whether the current state is solved."""

    def scramble(self, num_moves: int, rng: Random) -> tuple[StateKey, tuple[int, ...]]:
        """Scramble from a random solved state and return ``(state_key, actions)``.

        Default implementation shared by every puzzle: reset to a random solved
        state, then apply ``num_moves`` random non-backtracking actions. Actions
        undoing the previous move are avoided; if that leaves no choices (the
        action set can be state-dependent) the full action set is used instead.
        """
        assert num_moves >= 0, "num_moves must be non-negative"

        self.reset(rng.choice(self.solved_states()))
        actions: list[int] = []
        previous: int | None = None
        for _ in range(num_moves):
            candidates = [
                action
                for action in self.actions()
                if previous is None or action != self.inverse_action(previous)
            ]
            assert candidates, "too few actions to avoid backtracking"
            action = rng.choice(candidates)
            self.apply(action)
            actions.append(action)
            previous = action

        return self.state_key(), tuple(actions)
