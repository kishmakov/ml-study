#!/usr/bin/env python3

import ctypes
import random
import subprocess
import sys
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

    generator.generator_case_value.argtypes = [ctypes.c_size_t, ctypes.c_size_t, ctypes.c_char_p]
    generator.generator_case_value.restype = ctypes.c_bool
    return generator


def random_input(bitness: int) -> bytes:
    return "".join(random.choice("01") for _ in range(bitness)).encode("ascii")


def main() -> None:
    generator = load_generator()
    ser_id = 0
    bitness = generator.generator_get_input_bitness()
    cases_number = generator.generator_get_cases_number(ser_id)
    case_id = random.randrange(cases_number)

    print(f"ser_id: {ser_id}, case_id: {case_id}")
    for _ in range(5):
        input_value = random_input(bitness)
        output = generator.generator_case_value(ser_id, case_id, input_value)
        print(f"{input_value.decode('ascii')} -> {int(output)}")


if __name__ == "__main__":
    main()
