#include "generator.h"
#include "decision_tree.h"
#include "small_bitness.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <utility>

const size_t kCasesNumber = 1ull << 32; // some technical limitation

const size_t kMaxEffectiveSize = (1u << 16); // exclusive upper bound

namespace {

void ValidateBitness(uint16_t bitness) {
    assert(bitness <= kInputBitness);
}

uint32_t AllowedBits(uint16_t bitness) {
    ValidateBitness(bitness);
    if (bitness == kInputBitness) {
        return std::numeric_limits<uint32_t>::max();
    }
    return (1u << bitness) - 1;
}

bool RandomBool(std::mt19937& rng) {
    return std::uniform_int_distribution<int>(0, 1)(rng) != 0;
}

DecisionTree RandomTree(uint16_t bitness, std::mt19937& rng, size_t target_size, uint32_t allowed_bits) {
    DecisionTree tree(bitness);

    if (target_size == 0) { // Pure leaf tree
        tree.AddLeaf(RandomBool(rng));
        return tree;
    }

    tree.BuildSubtree(target_size, allowed_bits, /*used_bits=*/0, RandomBool(rng), rng);
    return tree;
}

inline std::mt19937 PrepRNG(uint16_t bitness, size_t case_id) {
    std::seed_seq seed{
        static_cast<uint32_t>(case_id),
        static_cast<uint32_t>(case_id >> 32),
        static_cast<uint32_t>(bitness),
    };
    return std::mt19937(seed);
}

inline size_t PrepSize(uint16_t bitness, size_t case_id, std::mt19937& rng) {
    (void)case_id;
    const size_t max_size = bitness >= 16
        ? kMaxEffectiveSize - 1
        : (1ull << bitness) - 1;
    return std::uniform_int_distribution<size_t>(0, max_size)(rng);
}

}  // namespace

const DecisionTree& GetRandomTree(uint16_t bitness, size_t case_id) {
    using CaseKey = std::pair<uint16_t, size_t>;

    static std::map<CaseKey, DecisionTree> generated_trees;
    static std::mutex generated_trees_mutex;

    ValidateBitness(bitness);

    std::lock_guard<std::mutex> lock(generated_trees_mutex);

    const CaseKey key{bitness, case_id};
    auto it = generated_trees.find(key);
    if (it == generated_trees.end()) {
        if (bitness <= kExactTableBitness) {
            it = generated_trees.emplace(key, SmallBitnessTree(bitness, case_id)).first;
        } else {
            std::mt19937 rng = PrepRNG(bitness, case_id);
            const size_t size = PrepSize(bitness, case_id, rng);
            it = generated_trees.emplace(key, RandomTree(bitness, rng, size, AllowedBits(bitness))).first;
        }
    }
    return it->second;
}

// API

size_t generator_get_input_bitness(void) {
    return kInputBitness;
}

size_t generator_get_cases_number(uint16_t bitness) {
    ValidateBitness(bitness);
    if (IsSmallBitness(bitness)) {
        return SmallBitnessCasesNumber(bitness);
    }
    return kCasesNumber;
}

size_t generator_case_nodes(uint16_t bitness, size_t case_id) {
    assert(case_id < generator_get_cases_number(bitness));

    const DecisionTree& tree = GetRandomTree(bitness, case_id);
    return tree.nodes.size() - tree.num_leafs;
}

const char* generator_case_active_bits(uint16_t bitness, size_t case_id) {
    assert(case_id < generator_get_cases_number(bitness));

    thread_local std::string active_bits;
    active_bits.assign(kInputBitness, '0');

    const DecisionTree& tree = GetRandomTree(bitness, case_id);
    for (size_t bit_id = 0; bit_id < tree.used_bits.size(); ++bit_id) {
        if (tree.used_bits[bit_id]) {
            active_bits[bit_id] = '1';
        }
    }

    return active_bits.c_str();
}

bool generator_case_value(uint16_t bitness, size_t case_id, const char* input) {
    assert(case_id < generator_get_cases_number(bitness));
    assert(input != nullptr);
    assert(std::strlen(input) == bitness);

    return GetRandomTree(bitness, case_id).Evaluate(std::string_view(input, bitness));
}
