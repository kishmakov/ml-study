#include <fstream>
#include <iomanip>

#include "math/metrics.h"
#include "Neuron.h"
#include "Weights.h"

namespace network {

[[maybe_unused]] inline std::string fullDstName(const std::string& scheme) { return "results/" + scheme + ".txt"; }

Weights Weights::loadFromFile(const std::string& baseName) {
    std::vector<double> weights;
    std::ifstream fin(fullDstName(baseName));

    double weight;

    while (fin >> weight) {
        weights.push_back(weight);
    }

    fin.close();

    return Weights(std::move(weights));
}

Weights& Weights::operator-=(const Weights& correction) {
    assert(size() == correction.size());

    for (size_t i = 0; i < size(); i++) {
        at(i) -= correction[i];
    }

    return *this;
}

Weights& Weights::operator+=(const Weights& correction) {
    assert(size() == correction.size());

    for (size_t i = 0; i < size(); i++) {
        at(i) += correction[i];
    }

    return *this;
}

Weights& Weights::operator*=(double mult) {
    for (auto& w: *this) {
        w *= mult;
    }

    return *this;
}

double Weights::distanceL2(const Weights& correction) const {
    return math::metricsL2(*this, correction);
}

void Weights::saveToFile(const std::string& scheme) {
    std::ofstream fout(fullDstName(scheme), std::ios::out);

    for (const auto& w: *this) {
        fout << std::setprecision(9) << w << std::endl;
    }

    fout.close();
}

} // network