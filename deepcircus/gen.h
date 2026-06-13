#pragma once

#include <cstddef>
#include <functional>
#include <iosfwd>
#include <string>
#include <vector>

using BooleanFunction = std::function<bool(const std::string&)>;

struct TestCase {
    std::string title;
    size_t inputBits;
    BooleanFunction func;
};

size_t GetTestsNumber();
TestCase GetTestById(size_t id);
