#include <cmath>

#include "metrics.h"

namespace math {

double metricsL2(const std::vector<double>& a, const std::vector<double>& b) {
    assert(a.size() == b.size());

    double SSE = 0;

    auto itA = a.cbegin();
    auto itB = b.cbegin();

    for (size_t iteration = a.size(); iteration > 0; --iteration) {
        double diff = *itA++ - *itB++;
        SSE += diff * diff;
    }

    SSE /= a.size();

    return std::sqrt(SSE);
}

} // math

