#ifndef CRUNCH_UTILITIES_H
#define CRUNCH_UTILITIES_H

#include "Case.h"
#include "network/Network.h"

double metricsMSE(const std::vector<Case>& cases, network::Network& net);

void correctionMSE(const std::vector<Case>& cases, network::Network& net);

#endif //CRUNCH_UTILITIES_H
