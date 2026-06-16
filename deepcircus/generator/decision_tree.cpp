#include "decision_tree.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

namespace {

void ValidateBitness(uint16_t bitness) {
    assert(bitness <= kInputBitness);
}

size_t CountBits(uint32_t bits) {
    return static_cast<size_t>(__builtin_popcount(bits));
}

size_t RandomUnusedBit(uint32_t allowed_bits, uint32_t used_bits, std::mt19937& rng) {
    const uint32_t free_mask = allowed_bits & ~used_bits;
    const size_t free_bits = CountBits(free_mask);
    assert(free_bits > 0);

    const size_t selected = std::uniform_int_distribution<size_t>(0, free_bits - 1)(rng);
    size_t seen = 0;
    for (size_t bit = 0; bit < kInputBitness; ++bit) {
        if ((free_mask & (1u << bit)) == 0) continue;
        if (seen == selected) return bit;
        ++seen;
    }
    assert(false);
    return 0;
}

bool RandomBool(std::mt19937& rng) {
    return std::uniform_int_distribution<int>(0, 1)(rng) != 0;
}

std::pair<size_t, size_t> SplitBudget(size_t n, std::mt19937& rng) {
    assert(n >= 1);
    const size_t remaining = n - 1;
    if (remaining == 0) return {0, 0};
    const size_t left = std::uniform_int_distribution<size_t>(0, remaining)(rng);
    return {left, remaining - left};
}

}  // namespace

DecisionTree::DecisionTree(uint16_t bitness)
    : used_bits(bitness, false)
    , bitness_(bitness)
{
    ValidateBitness(bitness_);
}

uint16_t DecisionTree::Bitness() const {
    return bitness_;
}

size_t DecisionTree::AddLeaf(bool value) {
    const size_t node_id = nodes.size();
    nodes.push_back(value);
    ++num_leafs;
    return node_id;
}

size_t DecisionTree::BuildSubtree(
    size_t budget,
    uint32_t allowed_bits,
    uint32_t path_used_bits,
    bool required_value,
    std::mt19937& rng)
{
    const size_t free_bits = CountBits(allowed_bits & ~path_used_bits);
    if (budget == 0 || free_bits == 0) {
        return AddLeaf(required_value);
    }

    const size_t node_id = nodes.size();
    nodes.push_back(false);

    const size_t bit_id = RandomUnusedBit(allowed_bits, path_used_bits, rng);
    used_bits[bit_id] = true;
    const uint32_t child_used_bits = path_used_bits | (1u << bit_id);

    auto [left_budget, right_budget] = SplitBudget(budget, rng);

    const size_t max_child_nodes = (1ull << (free_bits - 1)) - 1;
    left_budget = std::min(left_budget, max_child_nodes);
    right_budget = std::min(right_budget, max_child_nodes);

    const bool child0_required_value = RandomBool(rng);
    const bool child1_required_value = !child0_required_value;

    const size_t child0 = BuildSubtree(
        left_budget,
        allowed_bits,
        child_used_bits,
        child0_required_value,
        rng);
    const size_t child1 = BuildSubtree(
        right_budget,
        allowed_bits,
        child_used_bits,
        child1_required_value,
        rng);
    nodes[node_id] = Div{bit_id, child0, child1};
    return node_id;
}

bool DecisionTree::Evaluate(std::string_view input) const {
    assert(input.size() == bitness_);

    size_t node_id = 0;
    while (true) {
        const Node& node = nodes[node_id];
        const Div* division = std::get_if<Div>(&node);
        if (division == nullptr) {
            return std::get<bool>(node);
        }
        assert(division->bitId < input.size());
        node_id = input[division->bitId] == '1'
            ? division->child1
            : division->child0;
    }
}
