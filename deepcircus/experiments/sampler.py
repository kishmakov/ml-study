from __future__ import annotations

import numpy as np
import random

from experiments.generator import (
    Generator,
    generate_restriction_tensors as _generate_restriction_tensors,
    generate_value_tensors,
)


_SAMPLE_TENSOR_CACHE = {}
_RESTRICTION_TENSOR_CACHE = {}
_SAMPLE_TARGET_CACHE = {}


def _case_ids_key(case_ids) -> tuple[int, ...]:
    return tuple(int(case_id) for case_id in case_ids)


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
    processes: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    cache_key = (generator.library_path, bitness, _case_ids_key(case_ids), reps)
    if cache_key in _SAMPLE_TENSOR_CACHE:
        return _SAMPLE_TENSOR_CACHE[cache_key]

    print(f"Generating {len(case_ids)} sample tensors for bitness {bitness}")
    x = generate_value_tensors(generator, bitness, case_ids, reps, processes)

    # _SAMPLE_TENSOR_CACHE[cache_key] = x
    return x


def generate_restriction_tensors(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    reps: int,
    processes: int,
) -> np.ndarray:
    case_ids = list(case_ids)
    cache_key = (bitness, _case_ids_key(case_ids), reps)
    if cache_key in _RESTRICTION_TENSOR_CACHE:
        return _RESTRICTION_TENSOR_CACHE[cache_key]

    x = _generate_restriction_tensors(generator, bitness, case_ids, reps, processes)
    # _RESTRICTION_TENSOR_CACHE[cache_key] = x
    return x


def generate_samples(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    reps: int,
    processes: int,
) -> tuple[np.ndarray, np.ndarray]:
    case_ids = list(case_ids)
    cache_key = (generator.library_path, bitness, _case_ids_key(case_ids), reps)
    if cache_key in _SAMPLE_TARGET_CACHE:
        return _SAMPLE_TARGET_CACHE[cache_key]

    x = generate_sample_tensors(generator, bitness, case_ids, reps, processes)
    y = np.asarray(
        [generator.case_nodes(bitness, case_id) for case_id in case_ids],
        dtype=np.int64,
    )
    result = (x, y)
    _SAMPLE_TARGET_CACHE[cache_key] = result
    return result
