from __future__ import annotations

import atexit
import ctypes
import multiprocessing as mp
import numpy as np
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from tqdm import tqdm


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

        library.generator_parity_value.argtypes = [
            ctypes.c_uint16,
            ctypes.c_char_p,
        ]
        library.generator_parity_value.restype = ctypes.c_char_p

        library.generator_parity_restrictions.argtypes = [
            ctypes.c_uint16,
            ctypes.c_size_t,
        ]
        library.generator_parity_restrictions.restype = ctypes.c_char_p
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

    def case_values(
        self,
        bitness: int,
        case_id: int,
        input_bits: Sequence[str],
    ) -> np.ndarray:
        samples = np.empty(
            (len(input_bits), sample_point_dim(bitness)),
            dtype=np.float32,
        )
        for row_id, bits in enumerate(input_bits):
            samples[row_id] = self.case_value(bitness, case_id, bits)
        return samples

    def case_restrictions(self, bitness: int, case_id: int, rep: int) -> np.ndarray:
        value = self.library.generator_case_restrictions(
            bitness,
            case_id,
            rep,
        )
        point_dim = restriction_point_dim(bitness)
        signed = _ascii_bits_to_signed(value, bitness * 2 * point_dim)
        return signed.reshape(bitness * 2, point_dim)

    def parity_value(self, bitness: int, input_bits: str) -> np.ndarray:
        value = self.library.generator_parity_value(
            bitness,
            input_bits.encode("ascii"),
        )
        return _ascii_bits_to_signed(value, sample_point_dim(bitness))

    def parity_restrictions(self, bitness: int, rep: int) -> np.ndarray:
        value = self.library.generator_parity_restrictions(
            bitness,
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


def _init_worker(generator: Generator):
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = generator


def _evaluate_inputs(
    generator: Generator,
    bitness: int,
    case_id: int,
    input_bits: Sequence[str],
) -> np.ndarray:
    return generator.case_values(bitness, case_id, input_bits)


def _evaluate_depth(
    generator: Generator,
    bitness: int,
    case_id: int,
    _: None,
) -> np.float32:
    return np.float32(generator.case_depth(bitness, case_id))


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


def _worker(task):
    worker_id, processes, bitness, indexed_payloads, sample_fn = task
    results = []
    for row_id, case_id, payload in indexed_payloads:
        assert case_id % processes == worker_id
        samples = sample_fn(_WORKER_GENERATOR, bitness, case_id, payload)
        results.append((row_id, samples))
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
    payloads: Sequence,
    processes: int,
) -> Iterator[tuple[int, np.ndarray]]:
    # Each process owns the case_ids whose case_id % processes equals its worker
    # id, so the C++ tree caches built per process stay warm across calls.
    assert len(case_ids) == len(payloads)
    fleet = _get_fleet(generator, processes)
    buckets = [[] for _ in range(processes)]
    for row_id, (case_id, payload) in enumerate(zip(case_ids, payloads)):
        buckets[case_id % processes].append((row_id, case_id, payload))

    pending = []
    for worker_id, indexed_payloads in enumerate(buckets):
        if indexed_payloads:
            task = (worker_id, processes, bitness, indexed_payloads, sample_fn)
            pending.append(fleet[worker_id].apply_async(_worker, (task,)))

    for async_result in pending:
        yield from async_result.get()


def generate_value_tensors(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    input_bits: Sequence[Sequence[str]],
    processes: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    assert len(case_ids) == len(input_bits)
    assert input_bits, "empty input"
    reps = len(input_bits[0])
    assert all(len(case_input_bits) == reps for case_input_bits in input_bits)
    point_dim = sample_point_dim(bitness)
    x = np.empty((len(case_ids), reps, point_dim), dtype=np.float32)
    results = _dispatch(
        generator,
        _evaluate_inputs,
        bitness,
        case_ids,
        input_bits,
        processes,
    )
    for row_id, samples in tqdm(
        results,
        total=len(case_ids),
        desc=f"values b={bitness}",
    ):
        x[row_id] = samples
    return x


def generate_depths_tensors(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    processes: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    y = np.empty(len(case_ids), dtype=np.float32)
    results = _dispatch(
        generator,
        _evaluate_depth,
        bitness,
        case_ids,
        [None] * len(case_ids),
        processes,
    )
    for row_id, depth in tqdm(
        results,
        total=len(case_ids),
        desc=f"depths b={bitness}",
    ):
        y[row_id] = depth
    return y


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
        generator,
        _sample_restrictions,
        bitness,
        case_ids,
        [reps] * len(case_ids),
        processes,
    ):
        start = row_id * restrictions_per_case
        x[start : start + restrictions_per_case] = samples
    return x
