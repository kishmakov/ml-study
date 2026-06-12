#pragma once

#include <functional>
#include <string>

using BooleanFunction = std::function<bool(const std::string&)>;

BooleanFunction solve(int N, std::string values);
