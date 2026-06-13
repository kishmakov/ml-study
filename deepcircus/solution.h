#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

inline uint64_t operator>>(const std::string& input, size_t bit) {
    return static_cast<uint64_t>(input.at(bit) - '0');
}

struct Div {
    size_t bitId;

    size_t node0Id;
    size_t node1Id;
};

struct Node {
    std::optional<Div> division;
    bool value;
};

std::vector<Node> Solve(uint64_t N, const std::function<bool(const std::string&)>& func);
