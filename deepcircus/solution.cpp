#include "solution.h"

#include <utility>

BooleanFunction solve(int N, std::string values) {
    return [=](const std::string& input) {
        int id = 0;
        for (int i = 0; i < N; i++) {
            id += (input[i] - '0') << (N - 1 - i);
        }

        return values[id] == '1';
    };
}
