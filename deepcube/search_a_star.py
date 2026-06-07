"""Generic A* search."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from heapq import heappop, heappush
from math import inf

from puzzle import Puzzle, StateFloat, StateKey


BatchCostToGo = Callable[[list[StateFloat]], list[float]]


@dataclass(frozen=True, slots=True)
class SearchResult:
    solved: bool
    actions: tuple[int, ...]
    generated_states: int


def solve_a_star(
    puzzle: Puzzle,
    cost_to_go_batch: BatchCostToGo,
    weight: float,
    max_states: int,
    pop_batch_size: int,
) -> SearchResult:
    """Solve one puzzle state while batching heuristic calls.

    This follows the same shape as DeepCubeA's parallel weighted A*: pop several
    frontier nodes, expand them, filter already-seen children, evaluate the
    remaining child heuristics in one batch, then push them into the open heap.
    """

    assert weight >= 0, "path_cost_weight must be non-negative"
    assert max_states > 0, "max_generated_states must be positive"
    assert pop_batch_size > 0, "pop_batch_size must be positive"

    start = puzzle.state_key()
    open_heap: list[tuple[float, int, float, StateKey]] = []
    best_cost: dict[StateKey, float] = {start: 0.0}
    closed_cost: dict[StateKey, float] = {}
    parents: dict[StateKey, tuple[StateKey, int, float]] = {}

    push_count = 0
    start_heuristic = cost_to_go_batch([puzzle.cost_to_go_input()])[0]
    heappush(
        open_heap,
        (_priority(0.0, start_heuristic, weight), push_count, 0.0, start),
    )

    while open_heap:
        popped: list[tuple[float, StateKey]] = []
        while open_heap and len(popped) < pop_batch_size:
            _, _, path_cost, state = heappop(open_heap)

            if path_cost != best_cost.get(state, inf):
                continue
            if closed_cost.get(state, inf) <= path_cost:
                continue
            puzzle.reset(state)
            if puzzle.is_solved():
                return SearchResult(
                    solved=True,
                    actions=_reconstruct_actions(state, parents),
                    generated_states=len(best_cost),
                )
            popped.append((path_cost, state))

        if not popped:
            continue

        candidate_inputs: list[StateFloat] = []
        candidates: list[tuple[float, StateKey]] = []
        for path_cost, state in popped:
            closed_cost[state] = path_cost
            puzzle.reset(state)

            for action in puzzle.actions():
                puzzle.reset(state)
                child, transition_cost = puzzle.apply(action)
                assert transition_cost >= 0, "A* requires non-negative transition costs"

                child_path_cost = path_cost + transition_cost
                if child_path_cost >= best_cost.get(child, inf):
                    continue
                if child not in best_cost and len(best_cost) >= max_states:
                    continue

                best_cost[child] = child_path_cost
                parents[child] = (state, action, transition_cost)
                candidate_inputs.append(puzzle.cost_to_go_input())
                candidates.append((child_path_cost, child))

        if not candidates:
            continue

        child_heuristics = cost_to_go_batch(candidate_inputs)
        assert len(child_heuristics) == len(candidates), (
            "batched heuristic returned wrong result count"
        )
        for (child_path_cost, child), child_heuristic in zip(
            candidates,
            child_heuristics,
        ):
            push_count += 1
            heappush(
                open_heap,
                (
                    _priority(child_path_cost, child_heuristic, weight),
                    push_count,
                    child_path_cost,
                    child,
                ),
            )

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
