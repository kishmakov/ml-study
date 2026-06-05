#include <cmath>
#include <map>
#include <vector>

#include "activation_functions.h"

namespace math {

static double randomInRange(double min, double max) {
    return min + ((max - min) * rand()) / (RAND_MAX);
}

static double sigmoidFunciton(double x) {
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}

static double sigmoidDerivative(double x) {
    double ex = exp(-x);
    double denom = (1 + ex);
    return ex / (denom * denom);
}

static double sigmoidInit(size_t id) {
    return randomInRange(-5.0, 5.0);
}

static double reluFunciton(double x) {
    return x < 0.0 ? 0.0 : x;
}

static double reluDerivative(double x) {
    return x < 0.0 ? 0.0 : 1.0;
}

static double reluInit(size_t id) {
    return randomInRange(0.2, 0.5);
}

static double tanhFunciton(double x) {
    double ex = exp(2 * x);
    return (ex - 1) / (ex + 1);
}

static double tanhDerivative(double x) {
    double ex = exp(x);
    double sech = 2 / (ex + 1 / ex);
    return sech * sech;
}

static double tanhInit(size_t id) {
    return randomInRange(-5.0, 5.0);
}

static std::map<std::string, ActivationFunction> TaggedActivationFunctions = {
        {"sigmoid", {&sigmoidFunciton, &sigmoidDerivative, &sigmoidInit}},
        {"relu", {&reluFunciton, &reluDerivative, &reluInit}},
        {"tanh", {&tanhFunciton, &tanhDerivative, &tanhInit}},
};

const ActivationFunction& activationByName(const std::string& name) {
    if (!TaggedActivationFunctions.contains(name)) {
        throw std::out_of_range("No activation function named " + name);
    }

    return TaggedActivationFunctions[name];
}

static std::map<std::string, std::vector<std::string>> TaggedNeuronPacks = {
        {"5sig", {"sigmoid", "sigmoid", "sigmoid", "sigmoid", "sigmoid"}},
        {"5tan", {"tanh", "tanh", "tanh", "tanh", "tanh"}},
        {"4relu_1sig", {"relu", "relu", "relu", "relu", "sigmoid"}},
};

const std::vector<std::string>& packByName(const std::string& name) {
    if (!TaggedNeuronPacks.contains(name)) {
        throw std::out_of_range("No neuron pack named " + name);
    }

    return TaggedNeuronPacks[name];
}

} // math