from __future__ import annotations

import atexit
import ctypes
import multiprocessing as mp
import numpy as np
import random
from collections.abc import Callable, Iterator
from pathlib import Path


DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]
LIBRARY = DEEPCIRCUS_DIR / "build" / "libgenerator.so"

_WORKER_GENERATOR = None
_FLEET = None
_FLEET_KEY = None


def sample_point_dim(bitness: int) -> int:
    return 2 * bitness + 1


def restriction_point_dim(bitness: int) -> int:
    return sample_point_dim(bitness - 1)


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

        library.generator_case_depth.argtypes = [ctypes.c_uint16, ctypes.c_size_t]
        library.generator_case_depth.restype = ctypes.c_size_t

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

    def case_depth(self, bitness: int, case_id: int) -> int:
        return int(self.library.generator_case_depth(bitness, case_id))

    def case_active_bits(self, bitness: int, case_id: int) -> str:
        return self.library.generator_case_active_bits(bitness, case_id).decode("ascii")

    def case_value(self, bitness: int, case_id: int, input_bits: str) -> np.ndarray:
        value = self.library.generator_case_value(
            bitness,
            case_id,
            input_bits.encode("ascii"),
        )
        return _ascii_bits_to_signed(value, sample_point_dim(bitness))

    def case_restrictions(self, bitness: int, case_id: int, rep: int) -> np.ndarray:
        value = self.library.generator_case_restrictions(
            bitness,
            case_id,
            rep,
        )
        point_dim = restriction_point_dim(bitness)
        signed = _ascii_bits_to_signed(value, bitness * 2 * point_dim)
        return signed.reshape(bitness * 2, point_dim)


def load_generator() -> Generator:
    return Generator(LIBRARY)


def _ascii_bits_to_signed(value: bytes, expected_len: int) -> np.ndarray:
    assert len(value) == expected_len
    bits = np.frombuffer(value, dtype=np.uint8).astype(np.int8) - ord("0")
    return bits * 2 - 1


def _sample_value(
    generator: Generator,
    bitness: int,
    case_id: int,
    reps: int,
) -> np.ndarray:
    rng = random.Random((bitness << 32) + case_id)
    point_dim = sample_point_dim(bitness)
    samples = np.empty((reps, point_dim), dtype=np.float32)
    for rep_id in range(reps):
        input_bits = "".join(rng.choice("01") for _ in range(bitness))
        samples[rep_id] = generator.case_value(bitness, case_id, input_bits)
    return samples


def _sample_block_inversions(
    generator: Generator,
    bitness: int,
    case_id: int,
    reps: int,
    blocks: int = 7,
) -> np.ndarray:
    assert reps == (1 << blocks)

    rng = random.Random((bitness << 32) + case_id)
    base_input = [rng.choice("01") for _ in range(bitness)]
    bit_blocks = _split_bit_blocks(bitness, blocks)
    point_dim = sample_point_dim(bitness)
    samples = np.empty((reps, point_dim), dtype=np.float32)

    for mask in range(reps):
        input_bits = base_input.copy()
        for block_id, bit_ids in enumerate(bit_blocks):
            if ((mask >> block_id) & 1) == 0:
                continue
            for bit_id in bit_ids:
                input_bits[bit_id] = "0" if input_bits[bit_id] == "1" else "1"
        samples[mask] = generator.case_value(bitness, case_id, "".join(input_bits))

    return samples


def _split_bit_blocks(bitness: int, blocks: int) -> list[range]:
    base_size = bitness // blocks
    remainder = bitness % blocks
    result = []
    start = 0
    for block_id in range(blocks):
        size = base_size + (1 if block_id < remainder else 0)
        result.append(range(start, start + size))
        start += size
    return result


def _sample_restrictions(
    generator: Generator,
    bitness: int,
    case_id: int,
    reps: int,
) -> np.ndarray:
    point_dim = restriction_point_dim(bitness)
    samples = np.empty((bitness * 2, reps, point_dim), dtype=np.float32)
    for rep in range(reps):
        samples[:, rep, :] = generator.case_restrictions(bitness, case_id, rep)
    return samples


def _init_worker(generator: Generator):
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = generator


def _worker(task):
    worker_id, processes, bitness, indexed_case_ids, reps, sample_fn = task
    results = []
    for row_id, case_id in indexed_case_ids:
        assert case_id % processes == worker_id
        results.append(
            (row_id, sample_fn(_WORKER_GENERATOR, bitness, case_id, reps))
        )
    return results


def _get_fleet(generator: Generator, processes: int) -> list:
    # A persistent fleet of single-worker pools, one per residue class. Residue r
    # always runs on fleet[r], so each worker process loads libgenerator.so once
    # and the C++ tree caches it builds for its case_ids keep growing and stay
    # reused for the whole lifetime of the run, not just a single call.
    global _FLEET, _FLEET_KEY
    key = (generator.library_path, processes)
    if _FLEET_KEY != key:
        close_fleet()
        context = mp.get_context("fork")
        _FLEET = [
            context.Pool(processes=1, initializer=_init_worker, initargs=(generator,))
            for _ in range(processes)
        ]
        _FLEET_KEY = key
    return _FLEET


def close_fleet() -> None:
    global _FLEET, _FLEET_KEY
    if _FLEET is not None:
        for pool in _FLEET:
            pool.close()
        for pool in _FLEET:
            pool.join()
    _FLEET = None
    _FLEET_KEY = None


atexit.register(close_fleet)


def _dispatch(
    generator: Generator,
    sample_fn: Callable,
    bitness: int,
    case_ids: list[int],
    reps: int,
    processes: int,
) -> Iterator[tuple[int, np.ndarray]]:
    # Each process owns the case_ids whose case_id % processes equals its worker
    # id, so the C++ tree caches built per process stay warm across calls.
    fleet = _get_fleet(generator, processes)
    buckets = [[] for _ in range(processes)]
    for row_id, case_id in enumerate(case_ids):
        buckets[case_id % processes].append((row_id, case_id))

    pending = []
    for worker_id, indexed_case_ids in enumerate(buckets):
        if indexed_case_ids:
            task = (worker_id, processes, bitness, indexed_case_ids, reps, sample_fn)
            pending.append(fleet[worker_id].apply_async(_worker, (task,)))

    for async_result in pending:
        yield from async_result.get()


def generate_value_tensors(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    reps: int,
    processes: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    point_dim = sample_point_dim(bitness)
    x = np.empty((len(case_ids), reps, point_dim), dtype=np.float32)
    for row_id, samples in _dispatch(
        generator, _sample_value, bitness, case_ids, reps, processes
    ):
        x[row_id] = samples
    return x


def generate_restriction_tensors(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    reps: int,
    processes: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    point_dim = restriction_point_dim(bitness)
    restrictions_per_case = bitness * 2
    x = np.empty(
        (len(case_ids) * restrictions_per_case, reps, point_dim),
        dtype=np.float32,
    )
    for row_id, samples in _dispatch(
        generator, _sample_restrictions, bitness, case_ids, reps, processes
    ):
        start = row_id * restrictions_per_case
        x[start : start + restrictions_per_case] = samples
    return x
