// Generator module for boolean function truth tables.
// main.cpp owns command-line interaction.

#include <bits/stdc++.h>

#include "gen.h"

using namespace std;

static string truth_from_id(int N, uint64_t fid) {
    const size_t m = 1ull << N;
    string s;
    s.reserve(m);
    for (size_t i = 0; i < m; ++i) {
        char bit = ((fid >> i) & 1ull) ? '1' : '0';
        s.push_back(bit);
    }
    return s;
}

template <class F>
static string truth_from_func(int N, F&& f) {
    const size_t m = 1ull << N;
    string s;
    s.reserve(m);
    vector<int> bits(N, 0);
    for (size_t i = 0; i < m; ++i) {
        for (int j = 0; j < N; ++j) {
            int shift = N - 1 - j; // MSB-first in lexicographic order
            bits[j] = (int)((i >> shift) & 1ull);
        }
        bool out = f(bits);
        s.push_back(out ? '1' : '0');
    }
    return s;
}

static void add_case(vector<TestCase>& tests, string title, string bits) {
    tests.push_back(TestCase{std::move(title), std::move(bits)});
}

void write_case(ostream& out, const TestCase& test) {
    out << test.title << '\n';
    out << test.values << '\n';
}

vector<TestCase> make_tests() {
    vector<TestCase> tests;

    // 1) Exhaustive: all boolean functions for N = 1,2,3
    for (int N = 1; N <= 3; ++N) {
        const uint64_t m = 1ull << N;               // truth table length
        const uint64_t total = 1ull << m;           // number of functions
        for (uint64_t fid = 0; fid < total; ++fid) {
            string title = to_string(N) + "-bit function #" + to_string(fid);
            add_case(tests, title, truth_from_id(N, fid));
        }
    }

    // 2) Full 8-bit unsigned integer operations.
    // Inputs are ordered MSB-first: a7..a0 b7..b0.
    // Results are wrapped to 8 bits, as uint8_t arithmetic.
    auto read_u8 = [](const vector<int>& v, int offset) {
        uint8_t x = 0;
        for (int i = 0; i < 8; ++i) {
            x = static_cast<uint8_t>((x << 1) | v[offset + i]);
        }
        return x;
    };

    auto op8 = [&](auto g, int bit) {
        return truth_from_func(16, [&](const vector<int>& v) {
            const uint8_t a = read_u8(v, 0);
            const uint8_t b = read_u8(v, 8);
            const uint8_t result = g(a, b);
            return ((result >> bit) & 1u) != 0;
        });
    };

    auto add_op_bits = [&](const string& name, auto g) {
        add_case(tests, name + "8: last bit (bit0)", op8(g, 0));
        add_case(tests, name + "8: middle bit (bit4)", op8(g, 4));
        add_case(tests, name + "8: first bit (bit7)", op8(g, 7));
    };

    add_op_bits("add", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a + b); });
    add_op_bits("sub", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a - b); });
    add_op_bits("and", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a & b); });
    add_op_bits("or", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a | b); });
    add_op_bits("xor", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a ^ b); });
    add_op_bits("mul", [](uint8_t a, uint8_t b) { return static_cast<uint8_t>(a * b); });

    // 3) Random boolean tables for 16-bit inputs (N=16)
    auto random_truth = [](int N, uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int> bit01(0, 1);
        const size_t m = 1ull << N;
        string s;
        s.resize(m);
        for (size_t i = 0; i < m; ++i) s[i] = bit01(rng) ? '1' : '0';
        return s;
    };

    const int random16_count = 3; // a few sizable random functions
    for (int i = 1; i <= random16_count; ++i) {
        string title = "Random 16-bit function #" + to_string(i);
        add_case(tests, title, random_truth(16, 0xC0FFEEull + i));
    }

    return tests;
}
