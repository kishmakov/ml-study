#!/usr/bin/env python3

import ctypes
import sys
from pathlib import Path


DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]
GENERATOR_DIR = DEEPCIRCUS_DIR / "generator"
BUILD_DIR = DEEPCIRCUS_DIR / "build"
LIBRARY = BUILD_DIR / "libgenerator.so"

sys.path.insert(0, str(DEEPCIRCUS_DIR))

from experiments.experiment_pooling import run_experiment


def load_generator() -> ctypes.CDLL:
    generator = ctypes.CDLL(str(LIBRARY))

    generator.generator_get_input_bitness.argtypes = []
    generator.generator_get_input_bitness.restype = ctypes.c_size_t

    generator.generator_get_series_number.argtypes = []
    generator.generator_get_series_number.restype = ctypes.c_size_t

    generator.generator_get_cases_number.argtypes = [ctypes.c_size_t]
    generator.generator_get_cases_number.restype = ctypes.c_size_t

    generator.generator_case_nodes.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    generator.generator_case_nodes.restype = ctypes.c_size_t

    generator.generator_case_active_bits.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    generator.generator_case_active_bits.restype = ctypes.c_char_p

    generator.generator_case_value.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p]
    generator.generator_case_value.restype = ctypes.c_bool
    return generator


def main() -> None:
    generator = load_generator()
    run_experiment(generator)


if __name__ == "__main__":
    main()
