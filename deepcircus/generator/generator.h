#pragma once

#include <stddef.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

size_t generator_get_input_bitness(void);
size_t generator_get_series_number(void);
size_t generator_get_cases_number(size_t series_id);

// Computes number of nodes for given case
size_t generator_case_nodes(size_t series_id, size_t case_id);

// Computes function value for given case
bool generator_case_value(size_t series_id, size_t case_id, const char* input);

#ifdef __cplusplus
}
#endif
