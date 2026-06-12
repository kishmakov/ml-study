#include "solution.h"

#include <utility>

BooleanFunction solve(int N, std::string values) {
    (void)N;
    (void)values;

    return [](const std::string& input) {
        (void)input;
        return true;
    };
}
