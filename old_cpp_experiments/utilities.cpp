#include "utilities.h"
#include "network/Network.h"
#include "network/Weights.h"


double metricsMSE(const std::vector<Case>& cases, network::Network& net) {
    double SSE = 0;

    for (const auto& kase: cases) {
        double error = kase.getTarget() - net.react(kase.asInputs());
        SSE += error * error;
    }

    return SSE / double(cases.size());
}


void correctionMSE(const std::vector<Case>& cases, network::Network& net) {
    network::Weights correctionN(net.getWeights().size());

    for (const auto& kase: cases) {
        auto inputs = kase.asInputs();
        double actual = net.react(inputs);
        auto newC = net.backPropagation(actual - kase.getTarget());
        correctionN += newC;
    }

    correctionN *= -1.0 / double(cases.size());
    net += correctionN;
}