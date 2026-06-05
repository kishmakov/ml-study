#ifndef CRUNCH_WEIGHTS_H
#define CRUNCH_WEIGHTS_H

#include <vector>

namespace network {

struct Weights : std::vector<double> {
    explicit Weights(size_t size) : std::vector<double>(size, 0.0) {}
    explicit Weights(std::vector<double>&& weights) : std::vector<double>(std::move(weights)) {}

    static Weights loadFromFile(const std::string& baseName);

    Weights& operator-=(const Weights& correction);
    Weights& operator+=(const Weights& correction);
    Weights& operator*=(double mult);

    double distanceL2(const Weights& correction) const;

    void saveToFile(const std::string& scheme);
};

typedef std::unique_ptr<Weights> WeightsUP;

} // network

#endif //CRUNCH_WEIGHTS_H
