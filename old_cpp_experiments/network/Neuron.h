#ifndef CRUNCH_NEURON_H
#define CRUNCH_NEURON_H

#include <vector>

#include "math/activation_functions.h"

namespace network {

typedef double* WeightP;
typedef const double* InputP;
typedef std::pair<WeightP, InputP> WeightInput;

typedef std::vector<WeightInput> WeightInputs;

struct Neuron : WeightInputs {
    Neuron(const std::string& funcName, size_t size);

    void initWeights(double* weights);
    void initInputs(const std::vector<InputP>& inputs);

    void shuffle();
    void react();

    [[nodiscard]] std::vector<double> backPropagationWeights(double delta) const;

    [[nodiscard]] std::vector<double> backPropagationInputs(double delta) const;

    double value = 0;

private:
    double derivative = 0;
    const math::ActivationFunction& af_;
};

} // network

#endif //CRUNCH_NEURON_H
