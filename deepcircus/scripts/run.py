#!/usr/bin/env python3

import ctypes
import sys
from pathlib import Path


DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]
GENERATOR_DIR = DEEPCIRCUS_DIR / "generator"
BUILD_DIR = DEEPCIRCUS_DIR / "build"
LIBRARY = BUILD_DIR / "libgenerator.so"

sys.path.insert(0, str(DEEPCIRCUS_DIR))


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

        library.generator_get_input_bitness.argtypes = []
        library.generator_get_input_bitness.restype = ctypes.c_size_t

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

    def input_bitness(self) -> int:
        return int(self.library.generator_get_input_bitness())

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


def main() -> None:
    from experiments.experiment_pooling import run_experiment

    generator = load_generator()
    run_experiment(generator)


if __name__ == "__main__":
    main()
