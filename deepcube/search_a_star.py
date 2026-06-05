"""Generic A* search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf

from puzzle import Puzzle, StateFloat, StateKey


CostToGo = Callable[[StateFloat], float]


@dataclass(frozen=True, slots=True)
class SearchResult:
    solved: bool
    actions: tuple[int, ...]
    generated_states: int


def solve_a_star(
    puzzle: Puzzle,
    cost_to_go: CostToGo,
    weight: float,
    max_generated_states: int,
) -> SearchResult:
    """Solve one puzzle state with A*.

    The search starts from ``puzzle.state_key()``. ``cost_to_go`` is the heuristic
    approximation h(s). At most ``max_generated_states`` unique states are
    generated.
    """

    assert weight >= 0, "path_cost_weight must be non-negative"
    assert max_generated_states > 0, "max_generated_states must be positive"

    start = puzzle.state_key()
    open_heap: list[tuple[float, int, float, StateKey]] = []
    best_cost: dict[StateKey, float] = {start: 0.0}
    closed_cost: dict[StateKey, float] = {}
    parents: dict[StateKey, tuple[StateKey, int, float]] = {}

    push_count = 0
    start_priority = _priority(0.0, cost_to_go(puzzle.cost_to_go_input()), weight)
    heappush(
        open_heap,
        (start_priority, push_count, 0.0, start),
    )

    while open_heap:
        _, _, path_cost, state = heappop(open_heap)

        if path_cost != best_cost.get(state, inf):
            continue
        if closed_cost.get(state, inf) <= path_cost:
            continue
        puzzle.reset(state)
        if puzzle.is_solved():
            actions = _reconstruct_actions(state, parents)
            return SearchResult(
                solved=True,
                actions=actions,
                generated_states=len(best_cost),
            )

        closed_cost[state] = path_cost

        for action in puzzle.actions():
            puzzle.reset(state)
            child, transition_cost = puzzle.apply(action)
            assert transition_cost >= 0, "A* requires non-negative transition costs"

            child_path_cost = path_cost + transition_cost
            if child_path_cost >= best_cost.get(child, inf):
                continue
            if child not in best_cost and len(best_cost) >= max_generated_states:
                continue

            best_cost[child] = child_path_cost
            parents[child] = (state, action, transition_cost)
            push_count += 1
            child_priority = _priority(
                child_path_cost,
                cost_to_go(puzzle.cost_to_go_input()),
                weight,
            )
            heappush(open_heap, (child_priority, push_count, child_path_cost, child))

    return SearchResult(
        solved=False,
        actions=(),
        generated_states=len(best_cost),
    )


def _priority(path_cost: float, heuristic: float, weight: float) -> float:
    return weight * path_cost + heuristic


def _reconstruct_actions(
    goal: StateKey,
    parents: dict[StateKey, tuple[StateKey, int, float]],
) -> tuple[int, ...]:
    actions: list[int] = []
    current = goal

    while current in parents:
        parent, action, _ = parents[current]
        actions.append(action)
        current = parent

    actions.reverse()
    return tuple(actions)
