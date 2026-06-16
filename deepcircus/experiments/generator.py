import ctypes
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
from tqdm import tqdm


DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]
LIBRARY = DEEPCIRCUS_DIR / "build" / "libgenerator.so"

_WORKER_GENERATOR = None


class Generator:
    def __init__(self, library_path: Path):
        self.library_path = str(library_path)
        self._library = None

    def __getstate__(self):
        return {"library_path": self.library_path}

    def __setstate__(self, state):
        self.library_path = state["library_path"]
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


def _append_bits(points, input_bits: str, generator: Generator, bitness: int, case_id: int):
    point = []
    output_bit = generator.case_value(
        bitness,
        case_id,
        input_bits,
    )
    for bit in input_bits:
        _append_bit(point, bit == "1")

    _append_bit(point, output_bit)
    points.append(point)


def _sample_case(bitness: int, case_id: int, generator: Generator, reps: int):
    rng = random.Random((bitness << 32) + case_id)

    points = []
    for _ in range(reps):
        input_bits = "".join(rng.choice("01") for _ in range(bitness))
        _append_bits(points, input_bits, generator, bitness, case_id)

        flipped_bits = list(input_bits)
        for bit_id in range(bitness):
            flipped_bits[bit_id] = "0" if flipped_bits[bit_id] == "1" else "1"
            _append_bits(points, "".join(flipped_bits), generator, bitness, case_id)
            flipped_bits[bit_id] = input_bits[bit_id]

    return np.asarray(points, dtype=np.float32).ravel()


def _init_worker(generator):
    global _WORKER_GENERATOR
    _WORKER_GENERATOR = generator


def _sample_worker(task):
    row_id, bitness, case_id, reps = task
    return (
        row_id,
        _sample_case(bitness, case_id, _WORKER_GENERATOR, reps),
    )


def generate_samples(
    generator: Generator,
    bitness: int,
    *,
    num: int | None = None,
    case_ids: list[int] | None = None,
    seed: int | None = None,
    reps: int = 100,
    processes: int = 16,
    pool_chunksize: int = 1,
    split_name: str = "samples",
) -> np.ndarray:
    if case_ids is None:
        assert num is not None, "num must be provided when case_ids are omitted"
        assert seed is not None, "seed must be provided when case_ids are omitted"
        case_ids = random.Random(seed).sample(range(generator.cases_number(bitness)), num)
    else:
        case_ids = list(case_ids)
        if num is not None:
            assert num == len(case_ids), "num must match the number of case_ids"

    feature_size = reps * (bitness + 1) * (bitness + 1)
    x = np.empty((len(case_ids), feature_size), dtype=np.float32)

    def tasks():
        for row_id, case_id in enumerate(case_ids):
            yield row_id, bitness, case_id, reps

    context = mp.get_context("fork")
    with context.Pool(
        processes=processes,
        initializer=_init_worker,
        initargs=(generator,),
    ) as pool:
        print(
            f"Generating {len(case_ids)} {split_name} feature vectors "
            f"for bitness {bitness} with {processes} processes"
        )
        results = pool.imap_unordered(_sample_worker, tasks(), chunksize=pool_chunksize)
        for row_id, features in tqdm(
            results,
            total=len(case_ids),
            desc=f"{split_name} b={bitness}",
        ):
            x[row_id] = features

    return x
