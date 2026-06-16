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

inline std::mt19937 PrepRestrictionRNG(
    uint16_t bitness,
    size_t case_id,
    size_t fixed_bit_id,
    size_t fixed_bit_value,
    size_t rep)
{
    std::seed_seq seed{
        static_cast<uint32_t>(case_id),
        static_cast<uint32_t>(case_id >> 32),
        static_cast<uint32_t>(bitness),
        static_cast<uint32_t>(fixed_bit_id),
        static_cast<uint32_t>(fixed_bit_value),
        static_cast<uint32_t>(rep),
        static_cast<uint32_t>(rep >> 32),
    };
    return std::mt19937(seed);
}

inline size_t FullBitId(size_t restricted_bit_id, size_t fixed_bit_id) {
    return restricted_bit_id < fixed_bit_id
        ? restricted_bit_id
        : restricted_bit_id + 1;
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
    const auto write_point = [&](size_t point_id) {
        const size_t offset = point_id * point_size;
        for (size_t bit_id = 0; bit_id < bitness; ++bit_id) {
            value[offset + bit_id] = point_input[bit_id];
        }
        bool fv = tree.Evaluate(std::string_view(point_input.data(), bitness));
        value[offset + bitness] = fv ? '1' : '0';
    };

    write_point(0);
    for (size_t bit_id = 0; bit_id < bitness; ++bit_id) {
        point_input[bit_id] = point_input[bit_id] == '1' ? '0' : '1';
        write_point(bit_id + 1);
        point_input[bit_id] = input[bit_id];
    }
    return value.c_str();
}

const char* generator_case_restrictions(uint16_t bitness, size_t case_id, size_t rep) {
    assert(case_id < generator_get_cases_number(bitness));
    assert(bitness > 0);

    thread_local std::string value;
    thread_local std::string full_input;
    thread_local std::string restricted_input;

    const size_t restricted_bitness = bitness - 1;
    const size_t restricted_point_size = bitness;
    const size_t restricted_sample_size = restricted_point_size * restricted_point_size;
    value.assign(bitness * 2 * restricted_sample_size, '0');
    full_input.assign(bitness, '0');
    restricted_input.assign(restricted_bitness, '0');

    const DecisionTree& tree = GetRandomTree(bitness, case_id);
    const auto write_restricted_point = [&](
        size_t sample_offset,
        size_t point_id,
        size_t fixed_bit_id)
    {
        const size_t point_offset = sample_offset + point_id * restricted_point_size;
        for (size_t restricted_bit_id = 0; restricted_bit_id < restricted_bitness; ++restricted_bit_id) {
            const size_t full_bit_id = FullBitId(restricted_bit_id, fixed_bit_id);
            value[point_offset + restricted_bit_id] = full_input[full_bit_id];
        }
        value[point_offset + restricted_bitness] = tree.Evaluate(
            std::string_view(full_input.data(), bitness)) ? '1' : '0';
    };

    for (size_t fixed_bit_id = 0; fixed_bit_id < bitness; ++fixed_bit_id) {
        for (size_t fixed_bit_value = 0; fixed_bit_value <= 1; ++fixed_bit_value) {
            std::mt19937 rng = PrepRestrictionRNG(
                bitness,
                case_id,
                fixed_bit_id,
                fixed_bit_value,
                rep);

            const size_t restriction_id = fixed_bit_id * 2 + fixed_bit_value;
            const size_t sample_offset = restriction_id * restricted_sample_size;
            for (size_t restricted_bit_id = 0; restricted_bit_id < restricted_bitness; ++restricted_bit_id) {
                restricted_input[restricted_bit_id] = RandomBool(rng) ? '1' : '0';
            }
            for (size_t full_bit_id = 0; full_bit_id < bitness; ++full_bit_id) {
                full_input[full_bit_id] = full_bit_id == fixed_bit_id
                    ? static_cast<char>('0' + fixed_bit_value)
                    : restricted_input[full_bit_id - (full_bit_id > fixed_bit_id ? 1 : 0)];
            }

            write_restricted_point(sample_offset, 0, fixed_bit_id);
            for (size_t restricted_bit_id = 0; restricted_bit_id < restricted_bitness; ++restricted_bit_id) {
                const size_t full_bit_id = FullBitId(restricted_bit_id, fixed_bit_id);
                full_input[full_bit_id] = full_input[full_bit_id] == '1' ? '0' : '1';
                write_restricted_point(sample_offset, restricted_bit_id + 1, fixed_bit_id);
                full_input[full_bit_id] = restricted_input[restricted_bit_id];
            }
        }
    }

    return value.c_str();
}
