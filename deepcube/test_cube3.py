"""Smoke test for solving scrambled Cube3 states with A*."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from random import Random

from costtogo import CostToGo
from cube3 import Cube3
from puzzle import StateKey
from search_a_star import solve_a_star


NUM_CASES = 20
SCRAMBLE_SEED_LIMIT = 2**32
PATH_COST_WEIGHT = 1.0


@dataclass(frozen=True, slots=True)
class Cube3Case:
    state: StateKey
    scramble_depth: int
    scramble_seed: int
    scramble_actions: tuple[int, ...]


def generate_cases(max_depth: int, seed: int) -> list[Cube3Case]:
    rng = Random(seed)
    cases: list[Cube3Case] = []
    for scramble_depth in range(1, max_depth + 1):
        scramble_seed = rng.randrange(SCRAMBLE_SEED_LIMIT)
        cube = Cube3()
        state, scramble_actions = cube.scramble(scramble_depth, scramble_seed)
        cases.append(Cube3Case(state, scramble_depth, scramble_seed, scramble_actions))

    return cases


def check_valid(start_state: StateKey, actions: tuple[int, ...]) -> bool:
    cube = Cube3(start_state)
    for action in actions:
        cube.apply(action)
    return cube.is_solved()


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--max-states", type=int, default=5000)
    args = parser.parse_args()

    cases = generate_cases(args.depth, args.seed)
    solved_count = 0

    cube3 = Cube3()
    cost_to_go = CostToGo()

    for idx, case in enumerate(cases):
        cube3.reset(case.state)
        result = solve_a_star(cube3, cost_to_go, PATH_COST_WEIGHT, args.max_states)
        if result.solved:
            assert check_valid(case.state, result.actions), result
        solved_count += int(result.solved)

        print(
            f"{idx}: depth={case.scramble_depth} "
            f"solved={result.solved} "
            f"solution={len(result.actions)} "
            f"states={result.generated_states}",
        )

    print(f"solved_total={solved_count}/{len(cases)}")


if __name__ == "__main__":
    main()
