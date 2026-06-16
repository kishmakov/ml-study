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

// Computes position of bit in masked sequence
inline size_t FullBitId(size_t bit_id, size_t fixed_id) {
    return bit_id < fixed_id ? bit_id : bit_id + 1;
}

// Writes one sample into `value` at `sample_offset`: the base point built from
// `input` followed by one point per free bit with that bit flipped. Each
// point stores the free-bit coordinates and then the tree's value on the full
// input. `fixed_bit_id == bitness` means no bit is fixed (every bit is free);
// otherwise that bit is held constant and excluded from the coordinates.
void WriteFlipSample(
    std::string& value,
    size_t sample_offset,
    std::string& input,
    const DecisionTree& tree,
    size_t fixed_bit_id)
{
    const size_t bitness = input.size();
    const size_t free_bits = fixed_bit_id < bitness ? bitness - 1 : bitness;
    const size_t point_size = free_bits + 1;

    const auto write_point = [&](size_t point_id) {
        const size_t offset = sample_offset + point_id * point_size;
        for (size_t coord = 0; coord < free_bits; ++coord) {
            value[offset + coord] = input[FullBitId(coord, fixed_bit_id)];
        }
        value[offset + free_bits] = tree.Evaluate({input.data(), bitness}) ? '1' : '0';
    };

    write_point(0);
    for (size_t coord = 0; coord < free_bits; ++coord) {
        char& bit = input[FullBitId(coord, fixed_bit_id)];
        bit = bit == '1' ? '0' : '1';
        write_point(coord + 1);
        bit = bit == '1' ? '0' : '1';
    }
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

const char* generator_case_value(uint16_t bitness, size_t case_id, const char* input) {
    assert(case_id < generator_get_cases_number(bitness));
    assert(input != nullptr);
    assert(std::strlen(input) == bitness);

    thread_local std::string value;
    thread_local std::string point_input;

    const size_t point_size = bitness + 1;
    value.assign(point_size * point_size, '0');
    point_input.assign(input, bitness);

    const DecisionTree& tree = GetRandomTree(bitness, case_id);
    WriteFlipSample(value, /*sample_offset=*/0, point_input, tree, /*fixed_bit_id=*/bitness);
    return value.c_str();
}

const char* generator_case_restrictions(uint16_t bitness, size_t case_id, size_t rep) {
    assert(case_id < generator_get_cases_number(bitness));
    assert(bitness > 0);

    thread_local std::string value;
    thread_local std::string full_input;

    std::mt19937 rng = PrepRNG(bitness, case_id);

    const size_t restricted_bitness = bitness - 1;
    const size_t sample_size = bitness * bitness; // restricted_point_size^2
    value.assign(bitness * 2 * sample_size, '0');
    full_input.assign(bitness, '0');

    const DecisionTree& tree = GetRandomTree(bitness, case_id);

    for (size_t fixed_bit_id = 0; fixed_bit_id < bitness; ++fixed_bit_id) {
        for (size_t fixed_bit_value = 0; fixed_bit_value <= 1; ++fixed_bit_value) {
            // Pin the fixed bit and sample random values for the free bits.
            full_input[fixed_bit_id] = static_cast<char>('0' + fixed_bit_value);
            for (size_t coord = 0; coord < restricted_bitness; ++coord) {
                full_input[FullBitId(coord, fixed_bit_id)] = RandomBool(rng) ? '1' : '0';
            }

            const size_t restriction_id = fixed_bit_id * 2 + fixed_bit_value;
            WriteFlipSample(value, restriction_id * sample_size, full_input, tree, fixed_bit_id);
        }
    }

    return value.c_str();
}
