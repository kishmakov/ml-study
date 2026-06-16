#pragma once

#include "decision_tree.h"

bool IsSmallBitness(uint16_t bitness);
size_t SmallBitnessCasesNumber(uint16_t bitness);
DecisionTree SmallBitnessTree(uint16_t bitness, size_t case_id);
