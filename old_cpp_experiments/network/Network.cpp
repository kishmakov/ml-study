#include <fstream>
#include <sstream>
#include <memory>
#include <utility>

#include "Network.h"

namespace network {

const double BIAS_INPUT = 1.0;

WeightsUP Network::zeroedWeights(size_t size) {
    std::vector<double> weights(size, 0.0);
    return std::make_unique<Weights>(std::move(weights));
}

Network::Network(const std::string& scheme) {
    buildNeurons(scheme);
    bindWeights(zeroedWeights(weightsSize_));
}

Network::Network(const std::string& scheme, const Weights& weights) {
    buildNeurons(scheme);
    bindWeights(std::make_unique<network::Weights>(weights));
}

std::vector<double> Network::getNeuronWeights(size_t index) const {
    if (index >= getNeuronsNumber()) {
        throw std::out_of_range("Index out of range");
    }

    std::vector<double> result;
    result.reserve(neurons_[index].size());

    for (const auto& weightInput: neurons_[index]) {
        result.push_back(*weightInput.first);
    }

    return result;
}

void Network::bindWeights(WeightsUP weights) {
    assert(weights->size() == weightsSize_);
    weights_ = std::move(weights);

    size_t weightsOffset = 0;
    for (auto& neuron: neurons_) {
        neuron.initWeights(&weights_->at(weightsOffset));
        weightsOffset += neuron.size();
    }
}

void Network::shuffle() {
    for (auto& neuron: neurons_) {
        neuron.shuffle();
    }
}

double Network::react(const std::vector<double>& inputs) {
    assert(inputs.size() == inputs_.size());

    for (size_t id = 0; id < inputs.size(); ++id) {
        inputs_[id] = inputs[id];
    }

    for (auto& neuron: neurons_) {
        neuron.react();
    }

    return neurons_.back().value;
}

Weights Network::backPropagation(double delta) const {
    auto deltas = neurons_.back().backPropagationInputs(delta);
    deltas[getNeuronsNumber() - 1] = delta;

    std::vector<double> result;
    result.reserve(weightsSize_);

    size_t neuronId = 0;

    for (const auto& neuron: neurons_) {
        auto partial = neuron.backPropagationWeights(deltas[neuronId]);
        result.insert(result.end(), partial.begin(), partial.end());

        ++neuronId;
    }

    return Weights(std::move(result));
}

Network& Network::operator+=(const Weights& correction) {
    *weights_ += correction;
    return *this;
}

std::string fullSchemeName(const std::string& scheme) { return "schemes/" + scheme + ".txt"; }

typedef std::vector<int> NeuronInputs;

void Network::buildNeurons(const std::string& scheme) {
    size_t inputsSize = 0;
    weightsSize_ = 0;
    std::vector<NeuronInputs> neuronsInputs;

    std::ifstream networkAsFile(fullSchemeName(scheme));

    std::string neuronAsLine;
    while (std::getline(networkAsFile, neuronAsLine)) {
        std::istringstream neuronSerialized(neuronAsLine);
        std::string activationFunction;
        NeuronInputs neuronInputs;

        neuronSerialized >> activationFunction;

        int inputId;
        while (neuronSerialized >> inputId) {
            neuronInputs.push_back(inputId);
            if (inputId < 0) {
                inputsSize = std::max(inputsSize, size_t(-inputId));
            }
        }

        neurons_.emplace_back(activationFunction, neuronInputs.size() + 1);
        neuronsInputs.push_back(std::move(neuronInputs));
    }

    networkAsFile.close();

    inputs_.resize(inputsSize);

    for (size_t id = 0; id < neuronsInputs.size(); ++id) {
        std::vector<InputP> inputs;
        for (int num: neuronsInputs[id]) {
            inputs.push_back(num < 0 ? &inputs_[-1 - num] : &neurons_[num - 1].value);
        }

        inputs.push_back(&BIAS_INPUT);

        weightsSize_ += inputs.size();

        neurons_[id].initInputs(inputs);
    }
}

} // network