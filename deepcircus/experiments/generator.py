from __future__ import annotations

import ctypes
import numpy as np
import random
from collections.abc import Callable
from pathlib import Path
from tqdm import tqdm


DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]
LIBRARY = DEEPCIRCUS_DIR / "build" / "libgenerator.so"


class Generator:
    def __init__(self, library_path: Path):
        self.library_path = str(library_path)
        self._library = None

    @property
    def library(self) -> ctypes.CDLL:
        if self._library is None:
            self._library = self._load_library()
        return self._library

    def _load_library(self) -> ctypes.CDLL:
        library = ctypes.CDLL(self.library_path)

        library.generator_get_cases_number.argtypes = [ctypes.c_uint16]
        library.generator_get_cases_number.restype = ctypes.c_size_t

        library.generator_case_nodes.argtypes = [ctypes.c_uint16, ctypes.c_size_t]
        library.generator_case_nodes.restype = ctypes.c_size_t

        library.generator_case_active_bits.argtypes = [ctypes.c_uint16, ctypes.c_size_t]
        library.generator_case_active_bits.restype = ctypes.c_char_p

        library.generator_case_value.argtypes = [
            ctypes.c_uint16,
            ctypes.c_size_t,
            ctypes.c_char_p,
        ]
        library.generator_case_value.restype = ctypes.c_bool
        return library

    def cases_number(self, bitness: int) -> int:
        return int(self.library.generator_get_cases_number(bitness))

    def case_nodes(self, bitness: int, case_id: int) -> int:
        return int(self.library.generator_case_nodes(bitness, case_id))

    def case_active_bits(self, bitness: int, case_id: int) -> str:
        return self.library.generator_case_active_bits(bitness, case_id).decode("ascii")

    def case_value(self, bitness: int, case_id: int, input_bits: str) -> bool:
        return bool(
            self.library.generator_case_value(
                bitness,
                case_id,
                input_bits.encode("ascii"),
            )
        )


def load_generator() -> Generator:
    return Generator(LIBRARY)


def _append_bit(point: list[float], bit: bool):
    point.append(1.0 if bit else -1.0)


def _append_bits(points, input_bits: str, value_fn: Callable[[str], bool]):
    point = []
    output_bit = value_fn(input_bits)
    for bit in input_bits:
        _append_bit(point, bit == "1")

    _append_bit(point, output_bit)
    points.append(point)


def _sample_function(
    bitness: int,
    value_fn: Callable[[str], bool],
    reps: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    point_dim = (bitness + 1) * (bitness + 1)
    samples = np.empty((reps, point_dim), dtype=np.float32)
    for rep_id in range(reps):
        points = []
        input_bits = "".join(rng.choice("01") for _ in range(bitness))
        _append_bits(points, input_bits, value_fn)

        flipped_bits = list(input_bits)
        for bit_id in range(bitness):
            flipped_bits[bit_id] = "0" if flipped_bits[bit_id] == "1" else "1"
            _append_bits(points, "".join(flipped_bits), value_fn)
            flipped_bits[bit_id] = input_bits[bit_id]

        samples[rep_id] = np.asarray(points, dtype=np.float32).ravel()

    return samples


def _sample_case(bitness: int, case_id: int, generator: Generator, reps: int):
    return _sample_function(
        bitness,
        lambda input_bits: generator.case_value(bitness, case_id, input_bits),
        reps,
        seed=(bitness << 32) + case_id,
    )


def sample_restriction(
    generator: Generator,
    bitness: int,
    case_id: int,
    fixed_bit_id: int,
    fixed_bit_value: int,
    reps: int,
) -> np.ndarray:
    assert 0 <= fixed_bit_id < bitness
    assert fixed_bit_value in (0, 1)

    def value_fn(input_bits: str) -> bool:
        full_input_bits = (
            input_bits[:fixed_bit_id]
            + str(fixed_bit_value)
            + input_bits[fixed_bit_id:]
        )
        return generator.case_value(bitness, case_id, full_input_bits)

    seed = (
        (bitness << 48)
        + (case_id << 8)
        + (fixed_bit_id << 1)
        + fixed_bit_value
    )
    return _sample_function(bitness - 1, value_fn, reps, seed)


def generate_ids(
    generator: Generator,
    bitness: int,
    number: int,
    seed: int,
) -> list[int]:
    rng = random.Random(seed)
    return rng.sample(range(generator.cases_number(bitness)), number)


def generate_sample_tensors(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    reps: int,
) -> np.ndarray:
    case_ids = list(case_ids)

    point_dim = (bitness + 1) * (bitness + 1)
    x = np.empty((len(case_ids), reps, point_dim), dtype=np.float32)

    print(f"Generating {len(case_ids)} sample tensors for bitness {bitness}")
    for row_id, case_id in enumerate(tqdm(case_ids, desc=f"b={bitness}")):
        x[row_id] = _sample_case(bitness, case_id, generator, reps)

    return x


def generate_samples(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    reps: int,
) -> tuple[np.ndarray, np.ndarray]:
    case_ids = list(case_ids)
    x = generate_sample_tensors(generator, bitness, case_ids, reps)
    y = np.asarray(
        [generator.case_nodes(bitness, case_id) for case_id in case_ids],
        dtype=np.int64,
    )
    return x, y
