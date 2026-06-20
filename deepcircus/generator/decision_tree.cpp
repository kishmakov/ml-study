#include "decision_tree.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <utility>

namespace {

size_t RandomUnusedBit(const std::vector<bool>& path_used_bits, size_t free_bits, std::mt19937& rng) {
    assert(free_bits > 0);

    const size_t selected = std::uniform_int_distribution<size_t>(0, free_bits - 1)(rng);
    size_t seen = 0;
    for (size_t bit = 0; bit < path_used_bits.size(); ++bit) {
        if (path_used_bits[bit]) continue;
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

size_t MaxInternalNodes(size_t free_bits) {
    if (free_bits >= std::numeric_limits<size_t>::digits) {
        return std::numeric_limits<size_t>::max();
    }
    return (size_t{1} << free_bits) - 1;
}

size_t ComputeDepth(const std::vector<Node>& nodes, size_t node_id) {
    const Node& node = nodes[node_id];
    const Div* division = std::get_if<Div>(&node);
    if (division == nullptr) {
        return 0;
    }
    return 1 + std::max(
        ComputeDepth(nodes, division->child0),
        ComputeDepth(nodes, division->child1));
}

}  // namespace

DecisionTree::DecisionTree(uint16_t bitness)
    : used_bits(bitness, false)
    , bitness_(bitness)
{
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
    std::vector<bool>& path_used_bits,
    size_t path_used_count,
    bool required_value,
    std::mt19937& rng)
{
    assert(path_used_bits.size() == bitness_);
    assert(path_used_count <= bitness_);

    const size_t free_bits = bitness_ - path_used_count;
    if (budget == 0 || free_bits == 0) {
        return AddLeaf(required_value);
    }

    const size_t node_id = nodes.size();
    nodes.push_back(false);

    const size_t bit_id = RandomUnusedBit(path_used_bits, free_bits, rng);
    used_bits[bit_id] = true;
    path_used_bits[bit_id] = true;

    auto [left_budget, right_budget] = SplitBudget(budget, rng);

    const size_t max_child_nodes = MaxInternalNodes(free_bits - 1);
    left_budget = std::min(left_budget, max_child_nodes);
    right_budget = std::min(right_budget, max_child_nodes);

    const bool child0_required_value = RandomBool(rng);
    const bool child1_required_value = !child0_required_value;

    const size_t child0 = BuildSubtree(
        left_budget,
        path_used_bits,
        path_used_count + 1,
        child0_required_value,
        rng);
    const size_t child1 = BuildSubtree(
        right_budget,
        path_used_bits,
        path_used_count + 1,
        child1_required_value,
        rng);
    path_used_bits[bit_id] = false;
    nodes[node_id] = Div{bit_id, child0, child1};
    return node_id;
}

void DecisionTree::Finalize() {
    assert(!nodes.empty());
    depth = ComputeDepth(nodes, 0);
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
