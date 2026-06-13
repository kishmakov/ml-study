#include <bits/stdc++.h>

#include "gen.h"
#include "solution.h"

using namespace std;

static optional<int> bit_count_from_table_size(size_t size) {
    if (size == 0) return nullopt;

    int N = 0;
    size_t expected = 1;
    while (expected < size) {
        expected <<= 1;
        ++N;
    }
    if (expected != size) return nullopt;
    return N;
}

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

static void write_tests_file(const string& path, const vector<TestCase>& tests) {
    ofstream out(path, ios::binary);
    if (!out) {
        throw runtime_error("failed to open output file: " + path);
    }

    out << tests.size() << '\n';
    for (const TestCase& test : tests) {
        write_case(out, test);
    }
}

static size_t run_test(size_t test_id, const TestCase& test) {
    const optional<int> maybe_N = bit_count_from_table_size(test.values.size());
    if (!maybe_N) {
        cout << "FAIL #" << test_id << " " << test.title
             << ": truth table length is not a power of two\n";
        return 1;
    }

    const int N = *maybe_N;
    const BooleanFunction candidate = Solve(N, test.values);
    size_t local_failures = 0;
    string first_bad_input;
    char first_expected = '?';
    bool first_actual = false;

    for (size_t i = 0; i < test.values.size(); ++i) {
        const string input = input_for_index(N, i);
        const bool actual = candidate(input);
        const bool expected = test.values[i] == '1';
        if (actual != expected) {
            if (local_failures == 0) {
                first_bad_input = input;
                first_expected = test.values[i];
                first_actual = actual;
            }
            ++local_failures;
        }
    }

    if (local_failures == 0) {
        cout << "PASS #" << test_id << " " << test.title << "\n";
    } else {
        cout << "FAIL #" << test_id << " " << test.title
             << ": " << local_failures << " mismatches"
             << ", first at input " << first_bad_input
             << ", expected " << first_expected
             << ", got " << (first_actual ? '1' : '0') << "\n";
    }

    return local_failures;
}

static int run_tests(const vector<TestCase>& tests) {
    size_t failed_tests = 0;
    size_t failed_points = 0;

    for (size_t test_id = 0; test_id < tests.size(); ++test_id) {
        const size_t local_failures = run_test(test_id, tests[test_id]);
        if (local_failures != 0) ++failed_tests;
        failed_points += local_failures;
    }

    cout << "Summary: " << (tests.size() - failed_tests) << "/" << tests.size()
         << " tests passed";
    if (failed_tests != 0) {
        cout << ", " << failed_points << " point mismatches";
    }
    cout << "\n";

    return failed_tests == 0 ? 0 : 1;
}

static void print_usage(const char* argv0) {
    cerr << "Usage:\n"
         << "  " << argv0 << "                         runs all generated tests\n"
         << "  " << argv0 << " ID                      runs one generated test by zero-based id\n"
         << "  " << argv0 << " --id ID                 same as ID\n"
         << "  " << argv0 << " --count                  prints number of generated tests\n"
         << "  " << argv0 << " --write-input [PATH]     writes all generated tests, default input.txt\n";
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    const vector<TestCase> tests = make_tests();

    try {
        if (argc == 2 && string(argv[1]) == "--help") {
            print_usage(argv[0]);
            return 0;
        }

        if (argc == 2 && string(argv[1]) == "--count") {
            cout << tests.size() << '\n';
            return 0;
        }

        if ((argc == 2 && is_unsigned_integer(argv[1])) ||
            (argc == 3 && string(argv[1]) == "--id" && is_unsigned_integer(argv[2]))) {
            const string id_arg = argc == 2 ? argv[1] : argv[2];
            const size_t id = stoull(id_arg);
            if (id >= tests.size()) {
                cerr << "Test id " << id << " is out of range [0, " << tests.size() << ")\n";
                return 1;
            }
            const size_t failures = run_test(id, tests[id]);
            return failures == 0 ? 0 : 1;
        }

        if (argc == 2 && string(argv[1]) == "--write-input") {
            write_tests_file("input.txt", tests);
            cerr << "Wrote " << tests.size() << " cases to input.txt\n";
            return 0;
        }

        if (argc == 3 && string(argv[1]) == "--write-input") {
            write_tests_file(argv[2], tests);
            cerr << "Wrote " << tests.size() << " cases to " << argv[2] << "\n";
            return 0;
        }

        if (argc != 1) {
            print_usage(argv[0]);
            return 1;
        }

        return run_tests(make_tests());
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 2;
    }
}
