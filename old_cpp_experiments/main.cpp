#include <cmath>
#include <iostream>

#include "Case.h"
#include "math/metrics.h"
#include "network/Network.h"
#include "network/Weights.h"
#include "Plotter.h"
#include "network/Training.h"
#include "utilities.h"

static const size_t RAND_SEED = 20230402;
static const size_t CANDIDATES_NUMBER = 16;

const std::string MODE_TRAIN = "train";
const std::string MODE_CHECK = "check";


void plotWeightsAndMSE(const std::string& baseName,
                       const std::vector<Case>& cases,
                       network::Training& training) {
    Plot targetError("-b");
    Plot weightsDistance("-r");

    for (const auto& history: training.history) {
        network::Network net(training.scheme, *history);
        targetError += log10(metricsMSE(cases, net));
        weightsDistance += log10(training.result->distanceL2(*history));
    }

    Plotter plotter("Convergence on Training Set");

    plotter.add("MSE on train set", targetError);
    plotter.add("L_{2} weights", weightsDistance);

    plotter.xlabel = "Iteration / " + std::to_string(training.stepsPerReport);
    plotter.ylabel = "log_{10}";

    plotter.draw(baseName + "_mse_error");
}

// https://github.com/alandefreitas/matplotplusplus/blob/8dbea7d359f7b4f456bca7a6015c32b61ad728f4/source/matplot/util/colors.cpp
void plotNeurons(const std::string& baseName,
                 network::Training& training) {
    std::vector<Plot> plots = {
            Plot("-b"),
            Plot("-k"),
            Plot("-c"),
            Plot("-g"),
            Plot("-r")
    };

    network::Network resulting(training.scheme, *training.result);

    for (const auto& history: training.history) {
        network::Network historic(training.scheme, *history);

        for (size_t id = 0; id < resulting.getNeuronsNumber(); ++id) {
            plots[id] += log10(math::metricsL2(historic.getNeuronWeights(id), resulting.getNeuronWeights(id)));
        }
    }

    Plotter plotter("Neurons on Training Set");

    for (size_t id = 0; id < resulting.getNeuronsNumber(); ++id) {
        plotter.add("Neuron #" + std::to_string(id), plots[id]);
    }

    plotter.xlabel = "Iteration / " + std::to_string(training.stepsPerReport);
    plotter.ylabel = "log_{10} of L_{2}";

    plotter.draw(baseName + "_neurons");
}

void trainNetwork(std::string scheme, size_t stepsTotal, size_t stepsPerReport) {
    auto cases = Case::trainingSet();

    network::Training training(scheme, stepsPerReport);

    training.init(cases, CANDIDATES_NUMBER);
    training.run(cases, stepsTotal);
    training.result->saveToFile(scheme);

    plotWeightsAndMSE(scheme, cases, training);
    plotNeurons(scheme, training);
}

void checkNetwork(const std::string& functionName) {
    auto weights = network::Weights::loadFromFile(functionName);

    network::Network network(functionName, weights);

    unsigned score = 0;
    unsigned total = 0;

    for (const auto& kase: Case::trainingSet()) {
        total++;

        double actual = network.react(kase.asInputs());
        if (round(kase.getTarget()) == round(actual)) score++;
        for (unsigned i = 0; i < 4; ++i) {
            std::cout << kase.getInput(i) << " ";
        }
        std::cout << ": t=" << kase.getTarget();
        std::cout << " a=" << actual;
        std::cout << std::endl;
    }

    std::cout << score << " out of " << total << std::endl;
}


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error: not enough arguments." << std::endl;
        std::cerr << "Usage: crunch <mode> <scheme name>" << std::endl;
        return 1;
    }

    srand(RAND_SEED);

    std::string mode(argv[1]);
    std::string scheme(argv[2]);

    if (mode == MODE_TRAIN) {
        if (argc < 5) {
            std::cerr << "Error: not enough arguments." << std::endl;
            std::cerr << "Usage: crunch train <scheme name> <reports number> <steps per report>" << std::endl;
            return 1;
        }

        size_t reportsNumber = std::atol(argv[3]);
        size_t stepsPerReport = std::atol(argv[4]);

        trainNetwork(scheme, reportsNumber * stepsPerReport, stepsPerReport);
    }

    if (mode == MODE_CHECK) {
        checkNetwork(scheme);
    }

    return 0;
}

