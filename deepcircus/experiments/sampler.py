from __future__ import annotations

import numpy as np
import random
from pathlib import Path
import torch

from experiments.generator import (
    Generator,
    _sample_value,
    generate_restriction_tensors as _generate_restriction_tensors,
    generate_value_tensors,
)

torch.multiprocessing.set_sharing_strategy("file_system")


_SAMPLE_TENSOR_CACHE = {}
_RESTRICTION_TENSOR_CACHE = {}
_SAMPLE_TARGET_CACHE = {}


class DepthSampleDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        generator: Generator,
        bitness: int,
        case_ids: list[int],
        reps: int,
        *,
        shuffle: bool,
        seed: int,
    ):
        self.library_path = generator.library_path
        self.bitness = bitness
        self.case_ids = [int(case_id) for case_id in case_ids]
        self.reps = reps
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0
        self._generator = None

    def __len__(self) -> int:
        return len(self.case_ids)

    def __iter__(self):
        generator = self._get_generator()
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else worker.id
        workers = 1 if worker is None else worker.num_workers
        case_ids = [
            case_id for case_id in self.case_ids
            if case_id % workers == worker_id
        ]
        if self.shuffle:
            self._epoch += 1
            rng = random.Random(self.seed + self._epoch * workers + worker_id)
            rng.shuffle(case_ids)

        for case_id in case_ids:
            yield self._sample(generator, case_id)

    def _sample(self, generator: Generator, case_id: int) -> tuple[np.ndarray, np.float32]:
        x = _sample_value(generator, self.bitness, case_id, self.reps)
        y = np.float32(generator.case_depth(self.bitness, case_id))
        return x, y

    def _get_generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(Path(self.library_path))
        return self._generator


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


def make_depth_sample_loader(
    generator: Generator,
    bitness: int,
    case_ids: list[int],
    reps: int,
    batch_size: int,
    workers: int,
    *,
    shuffle: bool,
    drop_last: bool = False,
) -> torch.utils.data.DataLoader:
    dataset = DepthSampleDataset(
        generator,
        bitness,
        case_ids,
        reps,
        shuffle=shuffle,
        seed=bitness + len(case_ids),
    )
    kwargs = {}
    if workers > 0:
        kwargs = {
            "multiprocessing_context": "fork",
            "persistent_workers": True,
            "prefetch_factor": 4,
        }
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_numpy_batch,
        drop_last=drop_last,
        **kwargs,
    )


def _collate_numpy_batch(batch):
    x, y = zip(*batch)
    return np.stack(x).astype(np.float32), np.asarray(y, dtype=np.float32)


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
