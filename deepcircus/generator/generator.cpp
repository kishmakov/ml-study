#include "generator.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <random>
#include <string_view>
#include <vector>

const size_t kSeriesNumber = 2; // for now just one series of cases
const size_t kCasesNumber = 1ull << 32; // some technical limitation

const size_t kInputBitness = 32;
const size_t kMaxEffectiveSize = (1u << 16); // exclusive upper bound

struct Branch {
    std::optional<size_t> nodeId;
    bool value = false;
};

struct Div {
    size_t bitId;
    Branch branch0;
    Branch branch1;
};

struct Node {
    std::optional<Div> division;
    bool value = false;
};

size_t RandomUnusedBit(uint32_t used_bits, std::mt19937& rng) {
    const size_t free_bits = kInputBitness - static_cast<size_t>(__builtin_popcount(used_bits));
    const size_t selected = std::uniform_int_distribution<size_t>(0, free_bits - 1)(rng);
    size_t seen = 0;
    for (size_t bit = 0; bit < kInputBitness; ++bit) {
        if ((used_bits & (1u << bit)) != 0) continue;
         if (seen == selected) return bit;
        ++seen;
    }
    assert(false);
    return 0;
}

// Split a budget of (n-1) remaining nodes between two children.
// Left child gets k nodes, right child gets (n-1-k) nodes.
// k is drawn uniformly from [0, n-1] — this gives uniform distribution
// over all full binary tree shapes with n internal nodes (Rémy-inspired).
std::pair<size_t, size_t> SplitBudget(size_t n, std::mt19937& rng) {
    // n is the number of internal nodes for this subtree (>=1, since we're building one)
    // We spend 1 on the current node, leaving n-1 for children.
    assert(n >= 1);
    const size_t remaining = n - 1;
    if (remaining == 0) return {0, 0};
    const size_t left = std::uniform_int_distribution<size_t>(0, remaining)(rng);
    return {left, remaining - left};
}

Branch BuildBranch(
    size_t budget,          // number of internal nodes to use in this subtree
    uint32_t used_bits,
    std::vector<Node>& nodes,
    std::mt19937& rng);

size_t BuildNode(
    size_t budget,          // >= 1
    uint32_t used_bits,
    std::vector<Node>& nodes,
    std::mt19937& rng)
{
    const size_t node_id = nodes.size();
    nodes.push_back(Node{});

    const size_t free_bits = kInputBitness - static_cast<size_t>(__builtin_popcount(used_bits));

    // If we have no more bits to split on, this must become a leaf value
    // (we can't honour the budget, but correctness wins over budget)
    if (free_bits == 0) {
        std::uniform_int_distribution<int> bool_dist(0, 1);
        nodes[node_id].division = std::nullopt;
        nodes[node_id].value = bool_dist(rng) != 0;
        return node_id;
    }

    const size_t bit_id = RandomUnusedBit(used_bits, rng);
    const uint32_t child_used_bits = used_bits | (1u << bit_id);

    auto [left_budget, right_budget] = SplitBudget(budget, rng);

    // Cap child budgets by available bits on each path
    const size_t max_child_nodes = (1ull << free_bits) - 1; // conservative cap
    left_budget  = std::min(left_budget,  max_child_nodes);
    right_budget = std::min(right_budget, max_child_nodes);

    const Branch branch0 = BuildBranch(left_budget,  child_used_bits, nodes, rng);
    const Branch branch1 = BuildBranch(right_budget, child_used_bits, nodes, rng);
    nodes[node_id].division = Div{bit_id, branch0, branch1};
    return node_id;
}

Branch BuildBranch(
    size_t budget,
    uint32_t used_bits,
    std::vector<Node>& nodes,
    std::mt19937& rng)
{
    std::uniform_int_distribution<int> bool_dist(0, 1);
    if (budget == 0) {
        // Leaf
        return Branch{std::nullopt, bool_dist(rng) != 0};
    }
    return Branch{BuildNode(budget, used_bits, nodes, rng), false};
}

std::vector<Node> RandomTree(std::mt19937& rng, size_t target_size) {
    std::vector<Node> nodes;

    if (target_size == 0) { // Pure leaf tree
        std::uniform_int_distribution<int> bool_dist(0, 1);
        nodes.push_back(Node{std::nullopt, bool_dist(rng) != 0});
        return nodes;
    }

    BuildNode(target_size, /*used_bits=*/0, nodes, rng);
    return nodes;
}

bool EvaluateTree(const std::vector<Node>& nodes, std::string_view input) {
    size_t node_id = 0;
    while (true) {
        const Node& node = nodes[node_id];
        if (!node.division.has_value()) {
            return node.value;
        }
        const Branch& branch = input[node.division->bitId] == '1'
            ? node.division->branch1
            : node.division->branch0;
        if (!branch.nodeId.has_value()) {
            return branch.value;
        }
        node_id = *branch.nodeId;
    }
}

inline std::mt19937 PrepRNG(size_t series_id, size_t case_id) {
    return std::mt19937(case_id + (series_id << 30));
}

inline size_t PrepSize(size_t series_id, size_t case_id, std::mt19937& rng) {
    size_t low = series_id == 0 ? 0 : kMaxEffectiveSize - 10;
    size_t high = series_id == 0 ? 9 : kMaxEffectiveSize - 1;
    // Draw tree size uniformly from [low, high]
    return std::uniform_int_distribution<size_t>(low, high)(rng);
}

bool RandomTreeCase(size_t series_id, size_t case_id, std::string_view input) {
    std::mt19937 rng = PrepRNG(series_id, case_id);
    const size_t size = PrepSize(series_id, case_id, rng);
    return EvaluateTree(RandomTree(rng, size), input);
}

size_t RandomTreeNodes(size_t series_id, size_t case_id) {
    std::mt19937 rng = PrepRNG(series_id, case_id);
    return PrepSize(series_id, case_id, rng);
}

// API

size_t generator_get_input_bitness(void) {
    return kInputBitness;
}

size_t generator_get_series_number(void) {
    return kSeriesNumber;
}

size_t generator_get_cases_number(size_t series_id) {
    assert(series_id < kSeriesNumber);
    return kCasesNumber;
}

size_t generator_case_nodes(size_t series_id, size_t case_id) {
    assert(series_id < kSeriesNumber);
    assert(case_id < kCasesNumber);

    return RandomTreeNodes(series_id, case_id);
}

bool generator_case_value(size_t series_id, size_t case_id, const char* input) {
    assert(series_id < kSeriesNumber);
    assert(case_id < kCasesNumber);

    return RandomTreeCase(series_id, case_id, std::string_view(input, kInputBitness));
}
