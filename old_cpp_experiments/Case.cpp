#include "Case.h"

const uint64_t Case::BITNESS = 2;
const uint64_t Case::MASK = ((1 << BITNESS) - 1);

std::vector<Case> Case::trainingSet() {
    std::vector<Case> result;

    for (unsigned n1 = 0; n1 < 4; ++n1) {
        for (unsigned n2 = 0; n2 < 4; ++n2) {
            result.emplace_back(n1, n2);
        }
    }

    return result;
}

double Case::getInput(unsigned number) const {
    if (number >= 2 * BITNESS) {
        throw std::out_of_range("Index out of range");
    }

    return double((bits_ >> number) & 1);
}

std::vector<double> Case::asInputs() const {
    std::vector<double> result(2 * BITNESS, 0.0);

    for (unsigned i = 0; i < 2 * BITNESS; ++i) {
        result[i] = getInput(i);
    }

    return result;
}
