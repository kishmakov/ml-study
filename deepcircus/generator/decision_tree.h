#pragma once

#include <stddef.h>
#include <stdint.h>

#include <random>
#include <string_view>
#include <variant>
#include <vector>

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
    size_t depth = 0;

    size_t AddLeaf(bool value);

    size_t BuildSubtree(
        size_t budget,
        std::vector<bool>& path_used_bits,
        size_t path_used_count,
        bool required_value,
        std::mt19937& rng);

    void Finalize();

    bool Evaluate(std::string_view input) const;

private:
    uint16_t bitness_;
};
