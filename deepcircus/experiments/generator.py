from __future__ import annotations

import ctypes
import multiprocessing as mp
import numpy as np
import random
from collections.abc import Callable
from pathlib import Path
from tqdm import tqdm


DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]
LIBRARY = DEEPCIRCUS_DIR / "build" / "libgenerator.so"

_WORKER_GENERATOR = None


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
        library.generator_case_value.restype = ctypes.c_char_p

        library.generator_case_restrictions.argtypes = [
            ctypes.c_uint16,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
        library.generator_case_restrictions.restype = ctypes.c_char_p
        return library

    def cases_number(self, bitness: int) -> int:
        return int(self.library.generator_get_cases_number(bitness))

    def case_nodes(self, bitness: int, case_id: int) -> int:
        return int(self.library.generator_case_nodes(bitness, case_id))

    def case_active_bits(self, bitness: int, case_id: int) -> str:
        return self.library.generator_case_active_bits(bitness, case_id).decode("ascii")

    def case_value(self, bitness: int, case_id: int, input_bits: str) -> np.ndarray:
        value = self.library.generator_case_value(
            bitness,
            case_id,
            input_bits.encode("ascii"),
        )
        return _ascii_bits_to_signed(value, (bitness + 1) * (bitness + 1))

    def case_restrictions(self, bitness: int, case_id: int, rep: int) -> np.ndarray:
        value = self.library.generator_case_restrictions(
            bitness,
            case_id,
            rep,
        )
        point_dim = bitness * bitness
        signed = _ascii_bits_to_signed(value, bitness * 2 * point_dim)
        return signed.reshape(bitness * 2, point_dim)


def load_generator() -> Generator:
    return Generator(LIBRARY)


def _init_worker(generator: Generator):
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = generator


def _sample_restrictions_worker(task):
    worker_id, processes, bitness, indexed_case_ids, reps = task
    results = []
    for row_id, case_id in indexed_case_ids:
        assert case_id % processes == worker_id
        results.append(
            (
                row_id,
                sample_restrictions(
                    _WORKER_GENERATOR,
                    bitness,
                    case_id,
                    reps,
                ),
            )
        )
    return results


def make_restriction_pool(generator: Generator, processes: int):
    context = mp.get_context("fork")
    return context.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(generator,),
    )


def _ascii_bits_to_signed(value: bytes, expected_len: int) -> np.ndarray:
    assert len(value) == expected_len
    bits = np.frombuffer(value, dtype=np.uint8).astype(np.int8) - ord("0")
    return bits * 2 - 1


def _sample_function(
    bitness: int,
    value_fn: Callable[[str], np.ndarray],
    reps: int,
    seed: int,
) -> np.ndarray:
    rng = random.Random(seed)
    point_dim = (bitness + 1) * (bitness + 1)
    samples = np.empty((reps, point_dim), dtype=np.float32)
    for rep_id in range(reps):
        input_bits = "".join(rng.choice("01") for _ in range(bitness))
        samples[rep_id] = value_fn(input_bits)

    return samples


def _sample_case(bitness: int, case_id: int, generator: Generator, reps: int):
    return _sample_function(
        bitness,
        lambda input_bits: generator.case_value(bitness, case_id, input_bits),
        reps,
        seed=(bitness << 32) + case_id,
    )


def sample_restrictions(
    generator: Generator,
    bitness: int,
    case_id: int,
    reps: int,
) -> np.ndarray:
    point_dim = bitness * bitness
    samples = np.empty((bitness * 2, reps, point_dim), dtype=np.float32)
    for rep in range(reps):
        value = generator.case_restrictions(
            bitness,
            case_id,
            rep,
        )
        samples[:, rep, :] = value
    return samples


def generate_restriction_tensors(
    pool,
    bitness: int,
    case_ids: list[int],
    reps: int,
    processes: int,
) -> np.ndarray:
    point_dim = bitness * bitness
    restrictions_per_case = bitness * 2
    x = np.empty(
        (len(case_ids) * restrictions_per_case, reps, point_dim),
        dtype=np.float32,
    )

    buckets = [[] for _ in range(processes)]
    for row_id, case_id in enumerate(case_ids):
        buckets[case_id % processes].append((row_id, case_id))

    tasks = (
        (worker_id, processes, bitness, indexed_case_ids, reps)
        for worker_id, indexed_case_ids in enumerate(buckets)
        if indexed_case_ids
    )
    for results in pool.imap_unordered(_sample_restrictions_worker, tasks, chunksize=1):
        for row_id, samples in results:
            start = row_id * restrictions_per_case
            x[start : start + restrictions_per_case] = samples

    return x


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
