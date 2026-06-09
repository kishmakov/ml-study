#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace deepcube::puzzle {

class Environment {
public:
    virtual ~Environment() = default;

    virtual std::unique_ptr<Environment> getNextState(int action) const = 0;
    virtual std::vector<std::unique_ptr<Environment>> getNextStates() const = 0;
    virtual std::vector<uint8_t> getState() const = 0;
    virtual std::vector<float> costToGoInput() const = 0;
    virtual std::vector<int> getActions() const = 0;
    virtual bool isSolved() const = 0;
    virtual int getNumActions() const = 0;
};

std::unique_ptr<Environment> createEnvironment(
    const std::string& puzzle_name,
    const std::string& state_key
);

}  // namespace deepcube::puzzle
