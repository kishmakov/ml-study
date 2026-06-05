#ifndef CRUNCH_ACTIVATION_FUNCTIONS_H
#define CRUNCH_ACTIVATION_FUNCTIONS_H

#include <string>

namespace math {

typedef double (*DoubleFunction)(double);
typedef double (*InitFunction)(size_t);

struct ActivationFunction {
    DoubleFunction function;
    DoubleFunction derivative;
    InitFunction init;
};

const ActivationFunction& activationByName(const std::string& name);

const std::vector<std::string>& packByName(const std::string& name);

} // math

#endif //CRUNCH_ACTIVATION_FUNCTIONS_H

