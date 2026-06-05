#ifndef CRUNCH_CASE_H
#define CRUNCH_CASE_H

#include <cstdint>
#include <vector>

class Case {
    const static uint64_t BITNESS;
    const static uint64_t MASK;

    Case(uint64_t asBits, uint64_t m1, uint64_t m2) {
        bits_ = asBits;
        target_ = double((m1 * m2 >> (BITNESS - 1)) & 1);
    }

public:
    explicit Case(uint64_t asBits) : Case(asBits, MASK & asBits, MASK & (asBits >> BITNESS)) {}
    Case(uint64_t m1, uint64_t m2) : Case((m1 << BITNESS) | m2, m1, m2) {}

    static std::vector<Case> trainingSet();

    [[nodiscard]] inline double getTarget() const { return target_; }
    [[nodiscard]] double getInput(unsigned number) const;
    [[nodiscard]] std::vector<double> asInputs() const;

private:
    uint64_t bits_;
    double target_;
};

#endif //CRUNCH_CASE_H
