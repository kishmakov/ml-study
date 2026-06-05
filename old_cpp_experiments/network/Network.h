#ifndef CRUNCH_NETWORK_H
#define CRUNCH_NETWORK_H

#include "math/activation_functions.h"
#include "Neuron.h"
#include "Weights.h"

namespace network {

struct Network {
    static WeightsUP zeroedWeights(size_t size);

    explicit Network(const std::string& scheme);
    Network(const std::string& scheme, const Weights& weights);

    [[nodiscard]] size_t getNeuronsNumber() const {
        return neurons_.size();
    }

    [[nodiscard]]
    std::vector<double> getNeuronWeights(size_t index) const;

    [[nodiscard]]
    const Weights& getWeights() const { return *weights_; }

    void bindWeights(WeightsUP weights);
    void shuffle();
    double react(const std::vector<double>& inputs);

    [[nodiscard]]
    Weights backPropagation(double delta) const;

    Network& operator+=(const Weights& correction);

private:
    void buildNeurons(const std::string& scheme);

    size_t weightsSize_;
    WeightsUP weights_;
    std::vector<Neuron> neurons_;
    std::vector<double> inputs_;
};

} // network

#endif //CRUNCH_NETWORK_H
