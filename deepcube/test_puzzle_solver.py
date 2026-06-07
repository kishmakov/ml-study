"""Smoke test for solving scrambled puzzle states with A*."""

from __future__ import annotations

from argparse import ArgumentParser
from dataclasses import dataclass
from os.path import exists
from random import Random

from costtogo import CostToGo
from puzzle import Puzzle, StateKey
from puzzle_factory import DEFAULT_PUZZLE, PUZZLE_HELP, create_puzzle, model_path_for, load_cost_to_go
from search_a_star import solve_a_star


SCRAMBLE_SEED_LIMIT = 2**32
PATH_COST_WEIGHT = 0.1
POP_BATCH_SIZE = 64
INVERSE_CHECK_MIN_LENGTH = 10
INVERSE_CHECK_MAX_LENGTH = 20


@dataclass(frozen=True, slots=True)
class PuzzleCase:
    state: StateKey
    scramble_depth: int
    scramble_seed: int
    scramble_actions: tuple[int, ...]


def generate_cases(
    max_depth: int,
    seed: int,
    puzzle: Puzzle,
) -> list[PuzzleCase]:
    rng = Random(seed)
    cases: list[PuzzleCase] = []
    for scramble_depth in range(1, max_depth + 1):
        scramble_seed = rng.randrange(SCRAMBLE_SEED_LIMIT)
        puzzle.reset(puzzle.solved_states()[0])
        state, scramble_actions = puzzle.scramble(scramble_depth, Random(scramble_seed))
        cases.append(PuzzleCase(state, scramble_depth, scramble_seed, scramble_actions))

    return cases


def check_valid(
    start_state: StateKey,
    actions: tuple[int, ...],
    puzzle: Puzzle,
) -> bool:
    puzzle.reset(start_state)
    for action in actions:
        puzzle.apply(action)
    return puzzle.is_solved()


def check_random_inverse_sequences(seed: int, puzzle: Puzzle) -> None:
    rng = Random(seed)

    for length in range(INVERSE_CHECK_MIN_LENGTH, INVERSE_CHECK_MAX_LENGTH + 1):
        puzzle.reset(puzzle.solved_states()[0])
        initial_state = puzzle.state_key()
        actions: list[int] = []

        for _ in range(length):
            action = rng.choice(list(puzzle.actions()))
            puzzle.apply(action)
            actions.append(action)
        for action in reversed(actions):
            puzzle.apply(puzzle.inverse_action(action))

        assert puzzle.state_key() == initial_state, (length, actions)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--puzzle", default=DEFAULT_PUZZLE, help=PUZZLE_HELP)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=239)
    parser.add_argument("--max-states", type=int, default=5000)
    args = parser.parse_args()

    puzzle: Puzzle = create_puzzle(args.puzzle)
    check_random_inverse_sequences(args.seed, puzzle)

    cases = generate_cases(args.depth, args.seed, puzzle)
    solved_count = 0

    model_path = model_path_for(args.puzzle)
    if exists(model_path):
        cost_to_go = load_cost_to_go(puzzle, model_path, "cpu")
    else:
        cost_to_go = CostToGo()

    for idx, case in enumerate(cases):
        puzzle.reset(case.state)
        result = solve_a_star(
            puzzle,
            cost_to_go.batch,
            PATH_COST_WEIGHT,
            args.max_states,
            POP_BATCH_SIZE,
        )
        if result.solved:
            assert check_valid(case.state, result.actions, puzzle), result
        solved_count += int(result.solved)

        print(
            f"{idx}: depth={case.scramble_depth} "
            f"solved={1 if result.solved else 0} "
            f"solution={len(result.actions) if result.solved else '*'} "
            f"states={result.generated_states}",
        )

    print(f"solved_total={solved_count}/{len(cases)}")


if __name__ == "__main__":
    main()
