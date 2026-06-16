#pragma once

#include <stdint.h>
#include <stddef.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

size_t generator_get_cases_number(uint16_t bitness);

// Computes number of nodes for given case
size_t generator_case_nodes(uint16_t bitness, size_t case_id);

// Returns masked representation of active bits
const char* generator_case_active_bits(uint16_t bitness, size_t case_id);

// Computes function value for given case
bool generator_case_value(uint16_t bitness, size_t case_id, const char* input);

#ifdef __cplusplus
}
#endif
