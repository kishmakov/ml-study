#pragma once

#include <cstdint>
#include <functional>
#include <string>

using BooleanFunction = std::function<bool(const std::string&)>;

BooleanFunction Solve(uint64_t N, const std::string& values);
