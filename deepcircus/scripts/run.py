#!/usr/bin/env python3

import ctypes
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path


DEEPCIRCUS_DIR = Path(__file__).resolve().parents[1]
GENERATOR_DIR = DEEPCIRCUS_DIR / "generator"
BUILD_DIR = DEEPCIRCUS_DIR / "build"
LIBRARY = BUILD_DIR / "libgenerator.so"


def build_library() -> None:
    subprocess.run(
        [
            "cmake",
            "-S",
            str(GENERATOR_DIR),
            "-B",
            str(BUILD_DIR),
            "-DBUILD_SHARED_LIBS=ON",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        ["cmake", "--build", str(BUILD_DIR)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=sys.stderr,
    )


def load_generator() -> ctypes.CDLL:
    build_library()

    generator = ctypes.CDLL(str(LIBRARY))

    generator.generator_get_input_bitness.argtypes = []
    generator.generator_get_input_bitness.restype = ctypes.c_size_t

    generator.generator_get_cases_number.argtypes = [ctypes.c_size_t]
    generator.generator_get_cases_number.restype = ctypes.c_size_t

    generator.generator_case_nodes.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
    generator.generator_case_nodes.restype = ctypes.c_size_t

    generator.generator_case_value.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p]
    generator.generator_case_value.restype = ctypes.c_bool
    return generator


def main() -> None:
    generator = load_generator()

    ser_id = 0
    cases_number = generator.generator_get_cases_number(ser_id)
    node_sizes = [
        generator.generator_case_nodes(ser_id, random.randrange(cases_number))
        for _ in range(1024)
    ]
    histogram = Counter(node_sizes)

    print(f"series: {ser_id}")
    print(f"samples: {len(node_sizes)}")
    print(f"min_nodes: {min(node_sizes)}")
    print(f"max_nodes: {max(node_sizes)}")
    print(f"avg_nodes: {sum(node_sizes) / len(node_sizes):.2f}")
    print("histogram:")
    for nodes, count in sorted(histogram.items()):
        print(f"  {nodes}: {count}")


if __name__ == "__main__":
    main()
