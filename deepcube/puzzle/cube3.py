"""Rubik's Cube 3x3 description.

The state representation follows DeepCubeA's cube3 environment.  A cube is
described by the permutation and orientation of its 8 corners and 12 edges.
The solved state has corner_perm = (0..7), corner_ori = (0..0),
edge_perm = (0..11), edge_ori = (0..0).

Actions are integer indices into ``Cube3.action_names``:
``U-1, U1, D-1, D1, L-1, L1, R-1, R1, B-1, B1, F-1, F1``.

All moves are represented as wreath-product elements (Sym(n) ⋉ Zₖⁿ) and
composed directly on ``Cube3State``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from puzzle.puzzle import Puzzle, StateFloat, StateKey


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Cube3State:
    corner_perm: tuple[int, ...]   # 8 values in 0..7
    corner_ori:  tuple[int, ...]   # 8 values in 0..2
    edge_perm:   tuple[int, ...]   # 12 values in 0..11
    edge_ori:    tuple[int, ...]   # 12 values in 0..1


# ---------------------------------------------------------------------------
# Sticker-level constants  (used only at module load to build action tables)
# ---------------------------------------------------------------------------

_CUBE_LEN = 3
_NUM_STICKERS = 6 * _CUBE_LEN ** 2
_CENTER_STICKERS = tuple(face * 9 + 4 for face in range(6))

_CORNER_FACELETS: tuple[tuple[int, int, int], ...] = (
    (0, 26, 47), (2, 44, 20), (6, 53, 29), (8, 35, 38),
    (9, 18, 42), (11, 45, 24), (15, 36, 33), (17, 27, 51),
)
_EDGE_FACELETS: tuple[tuple[int, int], ...] = (
    (1, 23), (3, 50), (5, 41), (7, 32),
    (10, 21), (12, 39), (14, 48), (16, 30),
    (19, 43), (25, 46), (28, 52), (34, 37),
)

_CORNER_BY_FACELETS = {frozenset(f): i for i, f in enumerate(_CORNER_FACELETS)}
_EDGE_BY_FACELETS   = {frozenset(f): i for i, f in enumerate(_EDGE_FACELETS)}

_CORNER_BY_FACES = {
    frozenset(f // 9 for f in facelets): i
    for i, facelets in enumerate(_CORNER_FACELETS)
}
_EDGE_BY_FACES = {
    frozenset(f // 9 for f in facelets): i
    for i, facelets in enumerate(_EDGE_FACELETS)
}


# ---------------------------------------------------------------------------
# Sticker helpers  (used only at module load)
# ---------------------------------------------------------------------------

def _state_from_stickers(stickers: tuple[int, ...]) -> Cube3State:
    cp, co, ep, eo = [], [], [], []
    for facelets in _CORNER_FACELETS:
        observed = tuple(stickers[f] for f in facelets)
        piece = _CORNER_BY_FACELETS[frozenset(observed)]
        pf = _CORNER_FACELETS[piece]
        ori = next(o for o in range(3) if tuple(pf[(i + o) % 3] for i in range(3)) == observed)
        cp.append(piece); co.append(ori)
    for facelets in _EDGE_FACELETS:
        observed = tuple(stickers[f] for f in facelets)
        piece = _EDGE_BY_FACELETS[frozenset(observed)]
        pf = _EDGE_FACELETS[piece]
        ori = (pf, (pf[1], pf[0])).index(observed)
        ep.append(piece); eo.append(ori)
    return Cube3State(tuple(cp), tuple(co), tuple(ep), tuple(eo))


def _stickers_from_state(state: Cube3State) -> tuple[int, ...]:
    s: list[int] = [-1] * _NUM_STICKERS
    for c in _CENTER_STICKERS:
        s[c] = c
    for pos, facelets in enumerate(_CORNER_FACELETS):
        pf = _CORNER_FACELETS[state.corner_perm[pos]]; o = state.corner_ori[pos]
        for i, f in enumerate(facelets): s[f] = pf[(i + o) % 3]
    for pos, facelets in enumerate(_EDGE_FACELETS):
        pf = _EDGE_FACELETS[state.edge_perm[pos]]; o = state.edge_ori[pos]
        for i, f in enumerate(facelets): s[f] = pf[(i + o) % 2]
    return tuple(s)


# ---------------------------------------------------------------------------
# Rotation-index helpers  (used only to build action tables at module load)
# ---------------------------------------------------------------------------

def _flat(face: int, row: int, col: int) -> int:
    return face * 9 + row * 3 + col


def _items(v: int | range) -> tuple[int, ...]:
    return (v,) if isinstance(v, int) else tuple(v)


def _pairs(rs: int | range, cs: int | range) -> list[tuple[int, int]]:
    return [(r, c) for r in _items(rs) for c in _items(cs)]


_ADJ_FACES = {
    0: (2, 5, 3, 4), 1: (2, 4, 3, 5), 2: (0, 4, 1, 5),
    3: (0, 5, 1, 4), 4: (0, 3, 1, 2), 5: (0, 2, 1, 3),
}
_ADJ_IDXS: dict[int, dict[int, tuple]] = {
    0: {2: (range(3), 2), 3: (range(3), 2), 4: (range(3), 2), 5: (range(3), 2)},
    1: {2: (range(3), 0), 3: (range(3), 0), 4: (range(3), 0), 5: (range(3), 0)},
    2: {0: (0, range(3)), 1: (0, range(3)), 4: (2, range(2, -1, -1)), 5: (0, range(3))},
    3: {0: (2, range(3)), 1: (2, range(3)), 4: (0, range(2, -1, -1)), 5: (2, range(3))},
    4: {0: (range(3), 2), 1: (range(2, -1, -1), 0), 2: (0, range(3)), 3: (2, range(2, -1, -1))},
    5: {0: (range(3), 0), 1: (range(2, -1, -1), 2), 2: (2, range(3)), 3: (0, range(2, -1, -1))},
}
_CUBE_RING: tuple[tuple, ...] = (
    (0, range(3)), (range(3), 2), (2, range(2, -1, -1)), (range(2, -1, -1), 0),
)
_FACE_BY_NAME = {"U": 0, "D": 1, "L": 2, "R": 3, "B": 4, "F": 5}


def _compute_sticker_permutation(action_name: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return (new_idxs, old_idxs) defining the sticker permutation for one move."""
    face = _FACE_BY_NAME[action_name[0]]
    sign = int(action_name[1:])
    ft = _ADJ_FACES[face]
    if sign == 1:
        ff = ft[1:] + ft[:1]; cf = (3, 0, 1, 2)
    else:
        ff = ft[-1:] + ft[:-1]; cf = (1, 2, 3, 0)

    new_idxs: list[int] = []
    old_idxs: list[int] = []

    for ct, cfr in zip((0, 1, 2, 3), cf):
        for ni, oi in zip(_pairs(*_CUBE_RING[ct]), _pairs(*_CUBE_RING[cfr])):
            new_idxs.append(_flat(face, ni[0], ni[1]))
            old_idxs.append(_flat(face, oi[0], oi[1]))

    fi = _ADJ_IDXS[face]
    for fto, ffr in zip(ft, ff):
        for ni, oi in zip(_pairs(*fi[fto]), _pairs(*fi[ffr])):
            new_idxs.append(_flat(fto, ni[0], ni[1]))
            old_idxs.append(_flat(ffr, oi[0], oi[1]))

    return tuple(new_idxs), tuple(old_idxs)


def _apply_sticker_perm(
    stickers: tuple[int, ...],
    new_idxs: tuple[int, ...],
    old_idxs: tuple[int, ...],
) -> tuple[int, ...]:
    s = list(stickers)
    for ni, oi in zip(new_idxs, old_idxs):
        s[ni] = stickers[oi]
    return tuple(s)


# ---------------------------------------------------------------------------
# Build Cube3State-level action tables at module load
# ---------------------------------------------------------------------------

def _compute_action_state(action_name: str) -> Cube3State:
    """Return the Cube3State that results from applying *action_name* to the solved cube."""
    solved = tuple(range(_NUM_STICKERS))
    ni, oi = _compute_sticker_permutation(action_name)
    moved  = _apply_sticker_perm(solved, ni, oi)
    return _state_from_stickers(moved)


def _build_action_tables(
    action_names: tuple[str, ...],
) -> tuple[tuple[Cube3State, ...], ...]:
    """Pre-compute one Cube3State per action (the move applied to the solved cube)."""
    return tuple(_compute_action_state(name) for name in action_names)


# ---------------------------------------------------------------------------
# State serialisation / validation
# ---------------------------------------------------------------------------

def _state_to_key(state: Cube3State) -> StateKey:
    return "|".join(
        ",".join(str(v) for v in part)
        for part in (state.corner_perm, state.corner_ori, state.edge_perm, state.edge_ori)
    )


def _state_from_key(state_key: StateKey) -> Cube3State:
    parts = state_key.split("|")
    assert len(parts) == 4, "Cube3 state key must have four cubie fields"
    cp, co, ep, eo = (tuple(int(v) for v in p.split(",")) for p in parts)
    state = Cube3State(cp, co, ep, eo)
    _validate_state(state)
    return state


def _validate_state(state: Cube3State) -> None:
    assert len(state.corner_perm) == 8  and sorted(state.corner_perm) == list(range(8))
    assert len(state.corner_ori)  == 8  and all(0 <= o < 3 for o in state.corner_ori)
    assert len(state.edge_perm)   == 12 and sorted(state.edge_perm) == list(range(12))
    assert len(state.edge_ori)    == 12 and all(0 <= o < 2 for o in state.edge_ori)


# ---------------------------------------------------------------------------
# Solved-state enumeration  (24 orientations of a solved cube)
# ---------------------------------------------------------------------------

_FACE_VECTORS = ((0,1,0),(0,-1,0),(-1,0,0),(1,0,0),(0,0,-1),(0,0,1))
_FACE_BY_VECTOR = {v: f for f, v in enumerate(_FACE_VECTORS)}


def _dot(a: tuple[int,int,int], b: tuple[int,int,int]) -> int:
    return sum(x*y for x, y in zip(a, b))

def _cross(a: tuple[int,int,int], b: tuple[int,int,int]) -> tuple[int,int,int]:
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _rotation_face_perms() -> tuple[tuple[int, ...], ...]:
    perms: list[tuple[int, ...]] = []
    for up in _FACE_VECTORS:
        for front in _FACE_VECTORS:
            if _dot(up, front) != 0:
                continue
            right = _cross(up, front)
            perm = []
            for v in _FACE_VECTORS:
                rotated = (
                    v[0]*right[0] + v[1]*up[0] + v[2]*front[0],
                    v[0]*right[1] + v[1]*up[1] + v[2]*front[1],
                    v[0]*right[2] + v[1]*up[2] + v[2]*front[2],
                )
                perm.append(_FACE_BY_VECTOR[rotated])
            perms.append(tuple(perm))
    return tuple(dict.fromkeys(perms))


def _oriented_solved_state(face_perm: tuple[int, ...]) -> Cube3State:
    s = list(range(_NUM_STICKERS))
    for facelets, by_faces in ((_CORNER_FACELETS, _CORNER_BY_FACES),
                               (_EDGE_FACELETS,   _EDGE_BY_FACES)):
        for src_facelets in facelets:
            tgt_faces = frozenset(face_perm[f // 9] for f in src_facelets)
            tgt_facelets = facelets[by_faces[tgt_faces]]
            for sf in src_facelets:
                tgt_face = face_perm[sf // 9]
                tf = next(f for f in tgt_facelets if f // 9 == tgt_face)
                s[tf] = sf
    return _state_from_stickers(tuple(s))


def _compute_solved_states() -> tuple[Cube3State, ...]:
    states = list(dict.fromkeys(
        _oriented_solved_state(fp) for fp in _rotation_face_perms()
    ))
    assert len(states) == 24, f"Expected 24 solved orientations, got {len(states)}"
    return tuple(states)


# ---------------------------------------------------------------------------
# Module-level precomputed tables
# ---------------------------------------------------------------------------

_FACES        = ("U", "D", "L", "R", "B", "F")
_ACTION_NAMES = tuple(f"{face}{turn}" for face in _FACES for turn in (-1, 1))
_ACTION_STATES: tuple[Cube3State, ...] = _build_action_tables(_ACTION_NAMES)

_SOLVED_STATES      = _compute_solved_states()
_SOLVED_STATE_KEYS  = tuple(_state_to_key(s) for s in _SOLVED_STATES)
_SOLVED_STATE_SET   = frozenset(_SOLVED_STATES)


# ---------------------------------------------------------------------------
# Core state-transition: wreath-product composition
# ---------------------------------------------------------------------------

def _apply_action(state: Cube3State, action: int) -> Cube3State:
    """Apply *action* to *state* without touching any sticker arrays.

    Each action is stored as the Cube3State τ produced by that move on the
    solved cube.  Composing state σ with move τ in the wreath product gives:

        (σ · τ).corner_perm[p] = σ.corner_perm[ τ.corner_perm[p] ]
        (σ · τ).corner_ori[p]  = (σ.corner_ori[ τ.corner_perm[p] ] + τ.corner_ori[p]) % 3
    and analogously for edges (mod 2).
    """
    tau = _ACTION_STATES[action]
    return Cube3State(
        corner_perm = tuple(state.corner_perm[tau.corner_perm[p]] for p in range(8)),
        corner_ori  = tuple((state.corner_ori[tau.corner_perm[p]] + tau.corner_ori[p]) % 3
                            for p in range(8)),
        edge_perm   = tuple(state.edge_perm[tau.edge_perm[p]]     for p in range(12)),
        edge_ori    = tuple((state.edge_ori[tau.edge_perm[p]]     + tau.edge_ori[p]) % 2
                            for p in range(12)),
    )


# ---------------------------------------------------------------------------
# cost_to_go_input helper: face colors without building 54-sticker array
# ---------------------------------------------------------------------------

def _face_colors_from_state(state: Cube3State) -> np.ndarray:
    """Return a (54,) int64 array of face indices."""
    colors = np.empty(54, dtype=np.int64)

    # Centers are always their own face
    for c in _CENTER_STICKERS:
        colors[c] = c // 9

    for pos, facelets in enumerate(_CORNER_FACELETS):
        pf = _CORNER_FACELETS[state.corner_perm[pos]]
        o  = state.corner_ori[pos]
        for i, f in enumerate(facelets):
            colors[f] = pf[(i + o) % 3] // 9

    for pos, facelets in enumerate(_EDGE_FACELETS):
        pf = _EDGE_FACELETS[state.edge_perm[pos]]
        o  = state.edge_ori[pos]
        for i, f in enumerate(facelets):
            colors[f] = pf[(i + o) % 2] // 9

    return colors


# ---------------------------------------------------------------------------
# Cube3 class
# ---------------------------------------------------------------------------

class Cube3(Puzzle):
    """3x3 Rubik's Cube with the 12 DeepCubeA quarter-turn actions."""

    faces:          tuple[str, ...] = _FACES
    action_names:   tuple[str, ...] = _ACTION_NAMES
    inverse_actions: tuple[int, ...] = tuple(idx ^ 1 for idx in range(len(_ACTION_NAMES)))

    def __init__(self, state_key: StateKey | None = None) -> None:
        self._state = _state_from_key(state_key or _SOLVED_STATE_KEYS[0])

    # -- class-level helpers -------------------------------------------------

    @staticmethod
    def solved_states() -> list[StateKey]:
        return list(_SOLVED_STATE_KEYS)

    @staticmethod
    def inverse_action(action: int) -> int:
        return Cube3.inverse_actions[Cube3._validate_action(action)]

    # -- instance interface --------------------------------------------------

    def state_key(self) -> StateKey:
        return _state_to_key(self._state)

    def reset(self, state_key: StateKey) -> None:
        self._state = _state_from_key(state_key)

    def cost_to_go_input(self) -> StateFloat:
        return np.eye(6, dtype=np.float32)[_face_colors_from_state(self._state)].reshape(-1)

    def actions(self) -> range:
        return range(len(self.action_names))

    def apply(self, action: int) -> tuple[StateKey, float]:
        self._state = _apply_action(self._state, self._validate_action(action))
        return self.state_key(), 1.0

    def is_solved(self) -> bool:
        return self._state in _SOLVED_STATE_SET

    # -- private -------------------------------------------------------------

    @staticmethod
    def _validate_action(action: int) -> int:
        assert 0 <= action < len(Cube3.action_names), f"Invalid Cube3 action {action!r}"
        return action