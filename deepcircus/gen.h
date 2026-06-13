#pragma once

#include <iosfwd>
#include <string>
#include <vector>

struct TestCase {
    std::string title;
    std::string values;
};

std::vector<TestCase> make_tests();
