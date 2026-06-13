#include "generator.h"

#include <cassert>
#include <cstdint>
#include <string_view>
#include <vector>


const size_t kInputBitness = 32;
const size_t kSeriesNumber = 1;


bool FirstBit(std::string_view input) {
    return input[0] == '1';
}

bool LastBit(std::string_view input) {
    return input[kInputBitness - 1] == '1';
}

bool Parity(std::string_view input) {
    bool result = false;
    for (char bit : input) {
        result ^= bit == '1';
    }
    return result;
}

bool Majority(std::string_view input) {
    size_t ones = 0;
    for (char bit : input) {
        ones += bit == '1';
    }
    return ones >= (kInputBitness >> 1);
}

using CaseFunction = bool (*)(std::string_view);

const std::vector<CaseFunction> kCases = {
    FirstBit,
    LastBit,
    Parity,
    Majority,
};

// API

size_t generator_get_input_bitness(void) {
    return kInputBitness;
}

size_t generator_get_series_number(void) {
    return kSeriesNumber;
}

size_t generator_get_cases_number(size_t series_id) {
    assert(series_id < kSeriesNumber);
    return kCases.size();
}

bool generator_case_value(size_t series_id, size_t case_id, const char* input) {
    assert(series_id < kSeriesNumber);
    assert(case_id < kCases.size());

    return kCases[case_id](std::string_view(input, 32));
}
