#pragma once

#include <iosfwd>
#include <string>
#include <vector>

struct TestCase {
    std::string title;
    std::string values;
};

void write_case(std::ostream& out, const TestCase& test);
std::vector<TestCase> make_tests();
