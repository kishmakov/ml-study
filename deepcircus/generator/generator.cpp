#include "generator.h"
#include "decision_tree.h"
#include "small_bitness.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

const size_t kCasesNumber = 1ull << 32; // some technical limitation

const size_t kMaxEffectiveSize = (1u << 16); // exclusive upper bound

namespace {

bool RandomBool(std::mt19937& rng) {
    return std::uniform_int_distribution<int>(0, 1)(rng) != 0;
}

DecisionTree RandomTree(uint16_t bitness, std::mt19937& rng, size_t target_size) {
    DecisionTree tree(bitness);

    if (target_size == 0) { // Pure leaf tree
        tree.AddLeaf(RandomBool(rng));
        return tree;
    }

    std::vector<bool> path_used_bits(bitness, false);
    tree.BuildSubtree(
        target_size,
        path_used_bits,
        /*path_used_count=*/0,
        RandomBool(rng),
        rng);
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

inline size_t PrepSize(uint16_t bitness, std::mt19937& rng) {
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

    std::lock_guard<std::mutex> lock(generated_trees_mutex);

    const CaseKey key{bitness, case_id};
    auto it = generated_trees.find(key);
    if (it == generated_trees.end()) {
        if (bitness <= kExactTableBitness) {
            it = generated_trees.emplace(key, SmallBitnessTree(bitness, case_id)).first;
        } else {
            std::mt19937 rng = PrepRNG(bitness, case_id);
            const size_t size = PrepSize(bitness, rng);
            it = generated_trees.emplace(key, RandomTree(bitness, rng, size)).first;
        }
    }
    return it->second;
}

// API

size_t generator_get_cases_number(uint16_t bitness) {
    return IsSmallBitness(bitness)
        ? SmallBitnessCasesNumber(bitness)
        : kCasesNumber;
}

size_t generator_case_nodes(uint16_t bitness, size_t case_id) {
    assert(case_id < generator_get_cases_number(bitness));

    const DecisionTree& tree = GetRandomTree(bitness, case_id);
    return tree.nodes.size() - tree.num_leafs;
}

const char* generator_case_active_bits(uint16_t bitness, size_t case_id) {
    assert(case_id < generator_get_cases_number(bitness));

    thread_local std::string active_bits;
    const DecisionTree& tree = GetRandomTree(bitness, case_id);
    active_bits.assign(tree.used_bits.size(), '0');
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
