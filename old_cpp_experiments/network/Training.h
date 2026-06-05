#ifndef CRUNCH_TRAINING_H
#define CRUNCH_TRAINING_H

#include <string>

#include "Case.h"
#include "network/Network.h"
#include "network/Weights.h"

namespace network {

struct Training {
    Training(std::string scheme, size_t stepsPerReport);

    void init(const std::vector<Case>& cases, size_t candidatesNumber);
    void run(const std::vector<Case>& cases, size_t stepsTotal);

    Network net;

    std::string scheme;
    const size_t stepsPerReport;

    WeightsUP result;
    std::vector<WeightsUP> history;
};

} // network

#endif //CRUNCH_TRAINING_H
