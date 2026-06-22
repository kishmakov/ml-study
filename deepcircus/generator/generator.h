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

// Computes depth of the tree for given case
size_t generator_case_depth(uint16_t bitness, size_t case_id);

// Returns 0/1 masked representation of active bits
const char* generator_case_active_bits(uint16_t bitness, size_t case_id);

// Computes value of the function at input and all variations of input with one
// bit flipped.
// Return 0/1 string of length 2 * bitness + 1: input [bitness bits] +
// f(input) [1 bit] + f(input with flipped i-th bit) [1 x bitness bits]
const char* generator_case_value(uint16_t bitness, size_t case_id, const char* input);

// Computes all restriction sample points for one rep as ASCII 0/1 bytes
const char* generator_case_restrictions(uint16_t bitness, size_t case_id, size_t rep);

// Same as generator_case_value, but for parity function
const char* generator_parity_value(uint16_t bitness, const char* input);

// Same as generator_case_restrictions, but for parity function
const char* generator_parity_restrictions(uint16_t bitness, size_t rep);

#ifdef __cplusplus
}
#endif
