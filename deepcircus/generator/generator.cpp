#include "generator.h"

#include <cassert>
#include <cstdint>
#include <optional>
#include <random>
#include <string_view>
#include <vector>


const size_t kInputBitness = 32;
const size_t kSeriesNumber = 1;
const size_t kCasesNumber = 1ull << 32;
const size_t kMaxTreeDepth = kInputBitness;


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
        if ((used_bits & (1u << bit)) != 0) {
            continue;
        }
        if (seen == selected) {
            return bit;
        }
        ++seen;
    }

    assert(false);
    return 0;
}

Branch BuildRandomBranch(
    size_t depth,
    uint32_t used_bits,
    std::vector<Node>& nodes,
    std::mt19937& rng);

size_t BuildRandomDecisionNode(
    size_t depth,
    uint32_t used_bits,
    std::vector<Node>& nodes,
    std::mt19937& rng)
{
    const size_t node_id = nodes.size();
    nodes.push_back(Node{});

    const size_t bit_id = RandomUnusedBit(used_bits, rng);
    const uint32_t child_used_bits = used_bits | (1u << bit_id);
    const Branch branch0 = BuildRandomBranch(depth + 1, child_used_bits, nodes, rng);
    const Branch branch1 = BuildRandomBranch(depth + 1, child_used_bits, nodes, rng);
    nodes[node_id].division = Div{bit_id, branch0, branch1};
    return node_id;
}

Branch BuildRandomBranch(
    size_t depth,
    uint32_t used_bits,
    std::vector<Node>& nodes,
    std::mt19937& rng)
{
    std::uniform_int_distribution<int> bool_dist(0, 1);
    if (depth == kMaxTreeDepth || bool_dist(rng) == 0) {
        return Branch{std::nullopt, bool_dist(rng) != 0};
    }

    return Branch{BuildRandomDecisionNode(depth, used_bits, nodes, rng), false};
}

std::vector<Node> RandomTree(std::mt19937& rng) {
    std::vector<Node> nodes;
    const Branch root = BuildRandomBranch(0, 0, nodes, rng);
    if (!root.nodeId.has_value()) {
        nodes.push_back(Node{std::nullopt, root.value});
    }
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

bool RandomTreeCase(size_t case_id, std::string_view input) {
    std::mt19937 rng(static_cast<uint32_t>(case_id));
    return EvaluateTree(RandomTree(rng), input);
}

size_t RandomTreeNodes(size_t case_id) {
    std::mt19937 rng(static_cast<uint32_t>(case_id));
    return RandomTree(rng).size();
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

    return RandomTreeNodes(case_id);
}

bool generator_case_value(size_t series_id, size_t case_id, const char* input) {
    assert(series_id < kSeriesNumber);
    assert(case_id < kCasesNumber);

    return RandomTreeCase(case_id, std::string_view(input, kInputBitness));
}
