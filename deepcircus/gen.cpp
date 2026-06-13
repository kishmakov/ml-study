#include "gen.h"

#include <cstdint>
#include <stdexcept>
#include <utility>

namespace {

using Tests = std::vector<TestCase>;

uint64_t InputId(const std::string& input) {
    uint64_t id = 0;
    for (size_t bit = 0; bit < input.size(); ++bit) {
        if (input[bit] == '1') {
            id |= 1ull << bit;
        }
    }
    return id;
}

uint8_t ReadU8(const std::string& input, size_t offset) {
    uint8_t x = 0;
    for (size_t i = 0; i < 8; ++i) {
        const size_t bit = input.size() - 1 - (offset + i);
        x = static_cast<uint8_t>((x << 1) | static_cast<uint8_t>(input[bit] - '0'));
    }
    return x;
}

uint64_t Mix64(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

void AddCase(Tests& tests, std::string title, size_t inputBits, BooleanFunction func) {
    tests.push_back(TestCase{std::move(title), inputBits, std::move(func)});
}

template <class F>
void AddOpBits(Tests& tests, const std::string& name, F&& op) {
    auto opBit = [op](int bit) {
        return [op, bit](const std::string& input) {
            const uint8_t a = ReadU8(input, 0);
            const uint8_t b = ReadU8(input, 8);
            const uint8_t result = op(a, b);
            return ((result >> bit) & 1u) != 0;
        };
    };

    AddCase(tests, name + "8: last bit (bit0)", 16, opBit(0));
    AddCase(tests, name + "8: middle bit (bit4)", 16, opBit(4));
    AddCase(tests, name + "8: first bit (bit7)", 16, opBit(7));
}

Tests MakeTests() {
    Tests tests;

    for (size_t N = 1; N <= 3; ++N) {
        const uint64_t tableSize = 1ull << N;
        const uint64_t totalFunctions = 1ull << tableSize;
        for (uint64_t fid = 0; fid < totalFunctions; ++fid) {
            AddCase(
                tests,
                std::to_string(N) + "-bit function #" + std::to_string(fid),
                N,
                [N, fid](const std::string& input) {
                    const uint64_t id = InputId(input);
                    const uint64_t truthIndex = (1ull << N) - 1 - id;
                    return ((fid >> truthIndex) & 1ull) != 0;
                }
            );
        }
    }

    AddOpBits(tests, "add", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a + b); });
    AddOpBits(tests, "sub", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a - b); });
    AddOpBits(tests, "and", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a & b); });
    AddOpBits(tests, "or", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a | b); });
    AddOpBits(tests, "xor", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a ^ b); });
    AddOpBits(tests, "mul", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a * b); });

    for (int i = 1; i <= 3; ++i) {
        const uint64_t seed = 0xC0FFEEull + static_cast<uint64_t>(i);
        AddCase(
            tests,
            "Random 16-bit function #" + std::to_string(i),
            16,
            [seed](const std::string& input) {
                return (Mix64(seed ^ InputId(input)) & 1ull) != 0;
            }
        );
    }

    return tests;
}

const Tests& AllTests() {
    static const Tests tests = MakeTests();
    return tests;
}

}  // namespace

size_t GetTestsNumber() {
    return AllTests().size();
}

TestCase GetTestById(size_t id) {
    const Tests& tests = AllTests();
    if (id >= tests.size()) {
        throw std::out_of_range("test id is out of range");
    }
    return tests[id];
}
