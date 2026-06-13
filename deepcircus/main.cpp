#include <bits/stdc++.h>

#include "gen.h"
#include "solution.h"

using namespace std;

uint64_t no_optimisation_nodes = 0;
uint64_t total_nodes = 0;

static string input_for_index(uint64_t N, size_t index) {
    string input(N, '0');
    for (int bit = 0; bit < N; ++bit) {
        input[bit] = ((index >> bit) & 1) ? '1' : '0';
    }
    return input;
}

static bool is_unsigned_integer(const string& s) {
    if (s.empty()) return false;
    for (char c : s) {
        if (!isdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

static optional<uint64_t> input_count(uint64_t N) {
    if (N >= 63) return nullopt;
    return 1ull << N;
}

static size_t run_test(size_t test_id, const TestCase& test) {
    const uint64_t N = test.inputBits;
    const optional<uint64_t> maybe_input_count = input_count(N);
    if (!maybe_input_count) {
        cout << "FAIL #" << test_id << " " << test.title
             << ": input bit count is too large for the current bitmask representation\n";
        return 1;
    }

    const uint64_t total_inputs = *maybe_input_count;
    const auto nodes = Solve(N, test.func);

    auto candidate = [&](const std::string& input) {
        auto node = nodes[0];

        while (node.division) {
            bool bit = input >> node.division->bitId;
            node = nodes[bit ? node.division->node1Id : node.division->node0Id];
        }

        return node.value;
    };
    size_t local_failures = 0;
    string first_bad_input;
    bool first_expected = false;
    bool first_actual = false;

    for (size_t i = 0; i < total_inputs; ++i) {
        const string input = input_for_index(N, i);
        const bool actual = candidate(input);
        const bool expected = test.func(input);
        if (actual != expected) {
            if (local_failures == 0) {
                first_bad_input = input;
                first_expected = expected;
                first_actual = actual;
            }
            ++local_failures;
        }
    }

    if (local_failures == 0) {
        int decision_nodes = 0;
        for (auto& node: nodes) {
            decision_nodes += node.division ? 1 : 0;
        }
        no_optimisation_nodes += total_inputs - 1;
        total_nodes += decision_nodes;
        cout << "PASS #" << test_id << " " << test.title << " @ " << decision_nodes << " nodes \n";
    } else {
        cout << "FAIL #" << test_id << " " << test.title
             << ": " << local_failures << " mismatches"
             << ", first at input " << first_bad_input
             << ", expected " << (first_expected ? '1' : '0')
             << ", got " << (first_actual ? '1' : '0') << "\n";
    }

    std::cout << std::flush;

    return local_failures;
}

static int run_tests() {
    size_t failed_tests = 0;
    size_t failed_points = 0;
    const size_t tests_count = GetTestsNumber();

    for (size_t test_id = 0; test_id < tests_count; ++test_id) {
        const size_t local_failures = run_test(test_id, GetTestById(test_id));
        if (local_failures != 0) ++failed_tests;
        failed_points += local_failures;
    }

    cout << "Summary: " << (tests_count - failed_tests) << "/" << tests_count
         << " tests passed";
    if (failed_tests != 0) {
        cout << ", " << failed_points << " point mismatches";
    }
    cout << "\n";

    std::cout << "no_opt=" << no_optimisation_nodes
        << " cur=" << total_nodes << "\n";

    return failed_tests == 0 ? 0 : 1;
}

static void print_usage(const char* argv0) {
    cerr << "Usage:\n"
         << "  " << argv0 << "                         runs all generated tests\n"
         << "  " << argv0 << " ID                      runs one generated test by zero-based id\n"
         << "  " << argv0 << " --id ID                 same as ID\n"
         << "  " << argv0 << " --count                 prints number of generated tests\n";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    try {
        if (argc == 2 && string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        if (argc == 2 && string(argv[1]) == "--count") {
            cout << GetTestsNumber() << '\n';
            return 0;
        }

        if ((argc == 2 && is_unsigned_integer(argv[1])) ||
            (argc == 3 && string(argv[1]) == "--id" && is_unsigned_integer(argv[2]))) {
            const string id_arg = argc == 2 ? argv[1] : argv[2];
            const size_t id = stoull(id_arg);
            if (id >= GetTestsNumber()) {
                cerr << "Test id " << id << " is out of range [0, " << GetTestsNumber() << ")\n";
                return 1;
            }
            const size_t failures = run_test(id, GetTestById(id));
            return failures == 0 ? 0 : 1;
        }

        if (argc != 1) {
            print_usage(argv[0]);
            return 1;
        }

        return run_tests();
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
