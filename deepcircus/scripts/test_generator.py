import ctypes
from pathlib import Path


LIBRARY = Path(__file__).resolve().parents[1] / "build" / "libgenerator.so"

GOLDEN_VALUES = {
    (2, 3, "01"): "01001",
    (2, 15, "01"): "01111",
    (3, 31, "101"): "1010101",
    (3, 188, "110"): "1101101",
    (4, 3190, "0001"): "000100100",
    (4, 11304, "1101"): "110111001",
    (5, 3261348405, "11000"): "11000100011",
    (5, 390455940, "01001"): "01001010000",
    (6, 2547012052, "110111"): "1101110110110",
    (6, 883941716, "011111"): "0111110100001",
}


def load_library():
    library = ctypes.CDLL(str(LIBRARY))

    library.generator_get_cases_number.argtypes = [ctypes.c_uint16]
    library.generator_get_cases_number.restype = ctypes.c_size_t

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


def case_value(library, bitness, case_id, input_bits):
    return library.generator_case_value(
        bitness,
        case_id,
        input_bits.encode("ascii"),
    ).decode("ascii")


def test_case_value(library):
    print(f"Check generator_case_value ...")

    for (bitness, case_id, input_bits), expected in GOLDEN_VALUES.items():
        assert case_id < library.generator_get_cases_number(bitness)
        assert case_value(library, bitness, case_id, input_bits) == expected


def test_case_restrictions(library):
    print(f"Check generator_case_restrictions ...")

    for bitness, case_id, _ in GOLDEN_VALUES:
        free_bits = bitness - 1
        sample_size = 2 * free_bits + 1

        value = library.generator_case_restrictions(
            bitness,
            case_id,
            0,
        ).decode("ascii")
        assert len(value) == bitness * 2 * sample_size

        for fixed_bit_id in range(bitness):
            for fixed_bit_value in range(2):
                restriction_id = fixed_bit_id * 2 + fixed_bit_value
                offset = restriction_id * sample_size
                value_chunk = value[offset : offset + sample_size]

                full_input = list("0" * bitness)
                full_input[fixed_bit_id] = str(fixed_bit_value)
                for coord, bit_value in enumerate(value_chunk[:free_bits]):
                    full_bit_id = coord if coord < fixed_bit_id else coord + 1
                    full_input[full_bit_id] = bit_value
                full_input = "".join(full_input)

                direct_value = case_value(library, bitness, case_id, full_input)
                expected = (
                    value_chunk[:free_bits]
                    + direct_value[bitness]
                    + "".join(
                        direct_value[bitness + 1 + full_bit_id]
                        for full_bit_id in range(bitness)
                        if full_bit_id != fixed_bit_id
                    )
                )

                assert value_chunk == expected


if __name__ == "__main__":
    library = load_library()
    test_case_value(library)
    test_case_restrictions(library)