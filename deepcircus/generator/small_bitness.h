#pragma once

#include "decision_tree.h"

inline constexpr uint16_t kExactTableBitness = 4;

bool IsSmallBitness(uint16_t bitness);
size_t SmallBitnessCasesNumber(uint16_t bitness);
DecisionTree SmallBitnessTree(uint16_t bitness, size_t case_id);
