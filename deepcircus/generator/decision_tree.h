#pragma once

#include <stddef.h>
#include <stdint.h>

#include <random>
#include <string_view>
#include <variant>
#include <vector>

inline constexpr uint16_t kInputBitness = 32;
inline constexpr uint16_t kExactTableBitness = 4;

struct Div {
    size_t bitId;
    size_t child0;
    size_t child1;
};

using Node = std::variant<Div, bool>;

struct DecisionTree {
    explicit DecisionTree(uint16_t bitness);

    uint16_t Bitness() const;

    std::vector<Node> nodes;
    std::vector<bool> used_bits;
    size_t num_leafs = 0;

    size_t AddLeaf(bool value);

    size_t BuildSubtree(
        size_t budget,
        uint32_t allowed_bits,
        uint32_t path_used_bits,
        bool required_value,
        std::mt19937& rng);

    bool Evaluate(std::string_view input) const;

private:
    uint16_t bitness_;
};
