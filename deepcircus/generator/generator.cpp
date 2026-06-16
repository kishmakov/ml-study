#include "generator.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <map>
#include <mutex>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

const size_t kSeriesNumber = 2; // for now just one series of cases
const size_t kCasesNumber = 1ull << 32; // some technical limitation

const size_t kInputBitness = 32;
const size_t kMaxEffectiveSize = (1u << 16); // exclusive upper bound

struct Div {
    size_t bitId;
    size_t child0;
    size_t child1;
};

using Node = std::variant<Div, bool>;

struct DecisionTree {
    std::vector<Node> nodes;
    std::vector<size_t> used_bits;
    size_t num_leafs = 0;

    size_t AddLeaf(bool value) {
        const size_t node_id = nodes.size();
        nodes.push_back(value);
        ++num_leafs;
        return node_id;
    }

    size_t BuildSubtree(
        size_t budget,
        uint32_t path_used_bits,
        bool required_value,
        std::mt19937& rng);

    bool Evaluate(std::string_view input) const {
        size_t node_id = 0;
        while (true) {
            const Node& node = nodes[node_id];
            const Div* division = std::get_if<Div>(&node);
            if (division == nullptr) {
                return std::get<bool>(node);
            }
            node_id = input[division->bitId] == '1'
                ? division->child1
                : division->child0;
        }
    }
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

bool RandomBool(std::mt19937& rng) {
    return std::uniform_int_distribution<int>(0, 1)(rng) != 0;
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

size_t DecisionTree::BuildSubtree(
    size_t budget,          // number of internal nodes to use in this subtree
    uint32_t path_used_bits,
    bool required_value,    // at least one leaf in this subtree must have this value
    std::mt19937& rng)
{
    const size_t free_bits = kInputBitness - static_cast<size_t>(__builtin_popcount(path_used_bits));
    if (budget == 0 || free_bits == 0) {
        return AddLeaf(required_value);
    }

    const size_t node_id = nodes.size();
    nodes.push_back(false);

    const size_t bit_id = RandomUnusedBit(path_used_bits, rng);
    if (std::find(used_bits.begin(), used_bits.end(), bit_id) == used_bits.end()) {
        used_bits.push_back(bit_id);
    }
    const uint32_t child_used_bits = path_used_bits | (1u << bit_id);

    auto [left_budget, right_budget] = SplitBudget(budget, rng);

    // Cap child budgets by available bits on each path
    const size_t max_child_nodes = (1ull << (free_bits - 1)) - 1;
    left_budget  = std::min(left_budget,  max_child_nodes);
    right_budget = std::min(right_budget, max_child_nodes);

    // Force every generated split to have both output values somewhere below it.
    // This makes the tree syntactically non-constant under each internal node.
    const bool child0_required_value = RandomBool(rng);
    const bool child1_required_value = !child0_required_value;

    const size_t child0 = BuildSubtree(
        left_budget,
        child_used_bits,
        child0_required_value,
        rng);
    const size_t child1 = BuildSubtree(
        right_budget,
        child_used_bits,
        child1_required_value,
        rng);
    nodes[node_id] = Div{bit_id, child0, child1};
    return node_id;
}

DecisionTree RandomTree(std::mt19937& rng, size_t target_size) {
    DecisionTree tree;

    if (target_size == 0) { // Pure leaf tree
        tree.AddLeaf(RandomBool(rng));
        return tree;
    }

    tree.BuildSubtree(target_size, /*used_bits=*/0, RandomBool(rng), rng);
    return tree;
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

const DecisionTree& GetRandomTree(size_t series_id, size_t case_id) {
    using CaseKey = std::pair<size_t, size_t>;

    static std::map<CaseKey, DecisionTree> generated_trees;
    static std::mutex generated_trees_mutex;

    std::lock_guard<std::mutex> lock(generated_trees_mutex);

    const CaseKey key{series_id, case_id};
    auto it = generated_trees.find(key);
    if (it == generated_trees.end()) {
        std::mt19937 rng = PrepRNG(series_id, case_id);
        const size_t size = PrepSize(series_id, case_id, rng);
        it = generated_trees.emplace(key, RandomTree(rng, size)).first;
    }
    return it->second;
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

    const DecisionTree& tree = GetRandomTree(series_id, case_id);
    return tree.nodes.size() - tree.num_leafs;
}

const char* generator_case_active_bits(size_t series_id, size_t case_id) {
    assert(series_id < kSeriesNumber);
    assert(case_id < kCasesNumber);

    thread_local std::string active_bits;
    active_bits.assign(kInputBitness, '0');

    const DecisionTree& tree = GetRandomTree(series_id, case_id);
    for (size_t bit_id : tree.used_bits) {
        active_bits[bit_id] = '1';
    }

    return active_bits.c_str();
}

bool generator_case_value(size_t series_id, size_t case_id, const char* input) {
    assert(series_id < kSeriesNumber);
    assert(case_id < kCasesNumber);

    return GetRandomTree(series_id, case_id).Evaluate(std::string_view(input, kInputBitness));
}
