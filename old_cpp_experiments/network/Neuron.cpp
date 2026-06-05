#include "Neuron.h"

namespace network {

const static WeightInput NULLED(nullptr, nullptr);

Neuron::Neuron(const std::string& funcName, size_t size) : WeightInputs(size, NULLED),
    af_(math::activationByName(funcName))
{}

void Neuron::initWeights(double* weights) {
    for (auto& weightInput: *this) {
        weightInput.first = weights++;
    }
}

void Neuron::initInputs(const std::vector<InputP>& inputs) {
    assert(inputs.size() == size());

    auto inputsIterator = inputs.cbegin();

    for (auto& weightInput: *this) {
        weightInput.second = *inputsIterator++;
    }
}

void Neuron::shuffle() {
    for (size_t id = 0; id < size(); ++id) {
        *at(id).first = af_.init(id);
    }
}

void Neuron::react() {
    double sum = 0;

    for (auto& weightInput: *this) {
        sum += *weightInput.first * *weightInput.second;
    }

    value = af_.function(sum);
    derivative = af_.derivative(sum);
}

    std::vector<double> Neuron::backPropagationWeights(double delta) const {
    std::vector<double> result;
    result.reserve(size());

    delta *= derivative;
    for (auto& weightInput: *this) {
        result.push_back(delta * *weightInput.second);
    }

    return result;
}

std::vector<double> Neuron::backPropagationInputs(double delta) const {
    std::vector<double> result;
    result.reserve(size());

    delta *= derivative;
    for (auto& weightInput: *this) {
        result.push_back(delta * *weightInput.first);
    }

    return result;
}

} // network