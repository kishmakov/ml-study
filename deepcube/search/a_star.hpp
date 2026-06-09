#pragma once

#include "puzzle/environment.hpp"

#include <memory>
#include <vector>

namespace deepcube::search {

struct SearchResult {
    bool solved = false;
    std::vector<int> actions;
    int generated_states = 0;
};

class CostToGo {
public:
    virtual ~CostToGo() = default;

    virtual std::vector<float> batch(
        const std::vector<std::vector<float>>& states
    ) const = 0;
};

SearchResult aStarSearch(
    std::unique_ptr<puzzle::Environment> start,
    const CostToGo& cost_to_go,
    float weight,
    int max_states,
    int pop_batch_size
);

}  // namespace deepcube::search
