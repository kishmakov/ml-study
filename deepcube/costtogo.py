"""Cost-to-go approximations for search.

``cost_to_go(state_float) -> float``.
"""

from __future__ import annotations

from puzzle import StateFloat


class CostToGo:
    """Trivial heuristic that turns A* into uniform-cost search."""

    def __call__(self, state: StateFloat) -> float:
        return 0.0
