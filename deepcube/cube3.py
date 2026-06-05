"""Rubik's Cube 3x3 description.

The state representation follows DeepCubeA's cube3 environment: a cube is a
54-position sticker permutation. The solved state is ``(0, 1, ..., 53)``.
Actions are integer indices into ``Cube3.action_names``:

``U-1, U1, D-1, D1, L-1, L1, R-1, R1, B-1, B1, F-1, F1``.
"""

from __future__ import annotations

from random import Random

import numpy as np

from puzzle import StateFloat, StateKey


_CUBE_LEN = 3
_NUM_STICKERS = 6 * _CUBE_LEN * _CUBE_LEN
_SOLVED_STICKERS = tuple(range(_NUM_STICKERS))
_SOLVED_STATE_KEY = ",".join(str(sticker) for sticker in _SOLVED_STICKERS)


def _state_from_key(state_key: StateKey) -> tuple[int, ...]:
    stickers = tuple(int(sticker) for sticker in state_key.split(","))
    assert len(stickers) == _NUM_STICKERS, f"Cube3 state needs {_NUM_STICKERS} stickers, got {len(stickers)}"
    return stickers


def _state_to_key(stickers: tuple[int, ...]) -> StateKey:
    return ",".join(str(sticker) for sticker in stickers)


def _face_colors(stickers: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sticker // (_CUBE_LEN * _CUBE_LEN) for sticker in stickers)


class Cube3:
    """3x3 Rubik's Cube with the 12 DeepCubeA quarter-turn actions."""

    faces: tuple[str, ...] = ("U", "D", "L", "R", "B", "F")
    action_names: tuple[str, ...] = tuple(f"{face}{turn}" for face in faces for turn in (-1, 1))
    inverse_actions: tuple[int, ...] = tuple(idx ^ 1 for idx in range(len(action_names)))

    def __init__(self, state_key: StateKey = _SOLVED_STATE_KEY) -> None:
        self._rotate_idxs_new, self._rotate_idxs_old = self._compute_rotation_idxs()
        self._state = _state_from_key(state_key)

    @staticmethod
    def solved_state() -> StateKey:
        return _SOLVED_STATE_KEY

    @staticmethod
    def inverse_action(action: int) -> int:
        return Cube3.inverse_actions[Cube3._validate_action(action)]

    def state_key(self) -> StateKey:
        return _state_to_key(self._state)

    def reset(self, state_key: StateKey) -> None:
        self._state = _state_from_key(state_key)

    def cost_to_go_input(self) -> StateFloat:
        return np.asarray(_face_colors(self._state), dtype=np.float32)

    def actions(self) -> range:
        return range(len(self.action_names))

    def _next_stickers(self, stickers: tuple[int, ...], action: int) -> tuple[int, ...]:
        action = self._validate_action(action)
        stickers_next = list(stickers)

        for new_idx, old_idx in zip(self._rotate_idxs_new[action], self._rotate_idxs_old[action]):
            stickers_next[new_idx] = stickers[old_idx]

        return tuple(stickers_next)

    def apply(self, action: int) -> tuple[StateKey, float]:
        self._state = self._next_stickers(self._state, action)
        return self.state_key(), 1.0

    def is_solved(self) -> bool:
        return self._state == _SOLVED_STICKERS

    def scramble(self, num_moves: int, seed: int = 239) -> tuple[StateKey, tuple[int, ...]]:
        assert num_moves >= 0, "num_moves must be non-negative"

        rng = Random(seed)
        self._state = _SOLVED_STICKERS
        actions = []
        previous_action: int | None = None

        for _ in range(num_moves):
            candidates = list(self.actions())
            if previous_action is not None:
                candidates.remove(self.inverse_action(previous_action))
            action = rng.choice(candidates)
            self.apply(action)
            actions.append(action)
            previous_action = action

        return self.state_key(), tuple(actions)

    @staticmethod
    def _validate_action(action: int) -> int:
        assert 0 <= action < len(Cube3.action_names), f"Invalid Cube3 action {action!r}"
        return action

    @staticmethod
    def _flat_index(face: int, row: int, col: int) -> int:
        return face * _CUBE_LEN * _CUBE_LEN + row * _CUBE_LEN + col

    @staticmethod
    def _items(value: int | range) -> tuple[int, ...]:
        if isinstance(value, int):
            return (value,)
        return tuple(value)

    @classmethod
    def _pairs(cls, row_spec: int | range, col_spec: int | range) -> list[tuple[int, int]]:
        return [(row, col) for row in cls._items(row_spec) for col in cls._items(col_spec)]

    @classmethod
    def _compute_rotation_idxs(cls) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
        adj_faces = {
            0: (2, 5, 3, 4),
            1: (2, 4, 3, 5),
            2: (0, 4, 1, 5),
            3: (0, 5, 1, 4),
            4: (0, 3, 1, 2),
            5: (0, 2, 1, 3),
        }
        adj_idxs = {
            0: {
                2: (range(0, _CUBE_LEN), _CUBE_LEN - 1),
                3: (range(0, _CUBE_LEN), _CUBE_LEN - 1),
                4: (range(0, _CUBE_LEN), _CUBE_LEN - 1),
                5: (range(0, _CUBE_LEN), _CUBE_LEN - 1),
            },
            1: {
                2: (range(0, _CUBE_LEN), 0),
                3: (range(0, _CUBE_LEN), 0),
                4: (range(0, _CUBE_LEN), 0),
                5: (range(0, _CUBE_LEN), 0),
            },
            2: {
                0: (0, range(0, _CUBE_LEN)),
                1: (0, range(0, _CUBE_LEN)),
                4: (_CUBE_LEN - 1, range(_CUBE_LEN - 1, -1, -1)),
                5: (0, range(0, _CUBE_LEN)),
            },
            3: {
                0: (_CUBE_LEN - 1, range(0, _CUBE_LEN)),
                1: (_CUBE_LEN - 1, range(0, _CUBE_LEN)),
                4: (0, range(_CUBE_LEN - 1, -1, -1)),
                5: (_CUBE_LEN - 1, range(0, _CUBE_LEN)),
            },
            4: {
                0: (range(0, _CUBE_LEN), _CUBE_LEN - 1),
                1: (range(_CUBE_LEN - 1, -1, -1), 0),
                2: (0, range(0, _CUBE_LEN)),
                3: (_CUBE_LEN - 1, range(_CUBE_LEN - 1, -1, -1)),
            },
            5: {
                0: (range(0, _CUBE_LEN), 0),
                1: (range(_CUBE_LEN - 1, -1, -1), _CUBE_LEN - 1),
                2: (_CUBE_LEN - 1, range(0, _CUBE_LEN)),
                3: (0, range(_CUBE_LEN - 1, -1, -1)),
            },
        }
        face_by_name = {"U": 0, "D": 1, "L": 2, "R": 3, "B": 4, "F": 5}

        cube_idxs: tuple[tuple[int | range, int | range], ...] = (
            (0, range(0, _CUBE_LEN)),
            (range(0, _CUBE_LEN), _CUBE_LEN - 1),
            (_CUBE_LEN - 1, range(_CUBE_LEN - 1, -1, -1)),
            (range(_CUBE_LEN - 1, -1, -1), 0),
        )

        rotate_idxs_new: list[tuple[int, ...]] = []
        rotate_idxs_old: list[tuple[int, ...]] = []

        for move in cls.action_names:
            face_name = move[0]
            sign = int(move[1:])
            face = face_by_name[face_name]

            new_idxs: list[int] = []
            old_idxs: list[int] = []

            faces_to = adj_faces[face]
            if sign == 1:
                faces_from = faces_to[1:] + faces_to[:1]
                cubes_from = (3, 0, 1, 2)
            else:
                faces_from = faces_to[-1:] + faces_to[:-1]
                cubes_from = (1, 2, 3, 0)

            for cube_to, cube_from in zip((0, 1, 2, 3), cubes_from):
                for idx_new, idx_old in zip(cls._pairs(*cube_idxs[cube_to]), cls._pairs(*cube_idxs[cube_from])):
                    new_idxs.append(cls._flat_index(face, idx_new[0], idx_new[1]))
                    old_idxs.append(cls._flat_index(face, idx_old[0], idx_old[1]))

            face_idxs = adj_idxs[face]
            for face_to, face_from in zip(faces_to, faces_from):
                for idx_new, idx_old in zip(cls._pairs(*face_idxs[face_to]), cls._pairs(*face_idxs[face_from])):
                    new_idxs.append(cls._flat_index(face_to, idx_new[0], idx_new[1]))
                    old_idxs.append(cls._flat_index(face_from, idx_old[0], idx_old[1]))

            rotate_idxs_new.append(tuple(new_idxs))
            rotate_idxs_old.append(tuple(old_idxs))

        return tuple(rotate_idxs_new), tuple(rotate_idxs_old)
