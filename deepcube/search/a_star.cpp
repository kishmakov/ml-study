#include "search/a_star.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace deepcube::search {
namespace {

using StateKey = std::vector<uint8_t>;

struct StateHash {
    std::size_t operator()(const StateKey& state) const {
        std::size_t seed = state.size();
        for (uint8_t value : state) {
            seed ^= static_cast<std::size_t>(value) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        }
        return seed;
    }
};

struct Parent {
    StateKey state;
    int action = -1;
    float transition_cost = 1.0F;
};

struct OpenEntry {
    float priority = 0.0F;
    std::size_t push_count = 0;
    float path_cost = 0.0F;
    StateKey state;
    std::shared_ptr<const puzzle::Environment> env;
};

struct OpenEntryCompare {
    bool operator()(const OpenEntry& lhs, const OpenEntry& rhs) const {
        if (lhs.priority != rhs.priority) {
            return lhs.priority > rhs.priority;
        }
        return lhs.push_count > rhs.push_count;
    }
};

struct Candidate {
    float path_cost = 0.0F;
    StateKey state;
    std::shared_ptr<const puzzle::Environment> env;
};

float priority(float path_cost, float heuristic, float weight) {
    return weight * path_cost + heuristic;
}

std::vector<int> reconstructActions(
    const StateKey& goal,
    const std::unordered_map<StateKey, Parent, StateHash>& parents
) {
    std::vector<int> actions;
    StateKey current = goal;

    while (true) {
        const auto it = parents.find(current);
        if (it == parents.end()) {
            break;
        }

        actions.push_back(it->second.action);
        current = it->second.state;
    }

    std::reverse(actions.begin(), actions.end());
    return actions;
}

std::shared_ptr<const puzzle::Environment> toShared(
    std::unique_ptr<puzzle::Environment> env
) {
    return std::shared_ptr<const puzzle::Environment>(std::move(env));
}

}  // namespace

std::vector<float> DummyCostToGo::batch(
    const std::vector<std::vector<float>>& states
) const {
    return std::vector<float>(states.size(), 0.0F);
}

SearchResult aStarSearch(
    std::unique_ptr<puzzle::Environment> start,
    float weight,
    int max_states,
    int pop_batch_size
) {
    const DummyCostToGo dummy_cost_to_go;
    return aStarSearch(std::move(start), dummy_cost_to_go, weight, max_states, pop_batch_size);
}

SearchResult aStarSearch(
    std::unique_ptr<puzzle::Environment> start,
    const CostToGo& cost_to_go,
    float weight,
    int max_states,
    int pop_batch_size
) {
    if (start == nullptr) {
        throw std::runtime_error("A* start environment is null");
    }
    if (weight < 0.0F) {
        throw std::runtime_error("path cost weight must be non-negative");
    }
    if (max_states <= 0) {
        throw std::runtime_error("max generated states must be positive");
    }
    if (pop_batch_size <= 0) {
        throw std::runtime_error("pop batch size must be positive");
    }

    std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryCompare> open;
    std::unordered_map<StateKey, float, StateHash> best_cost;
    std::unordered_map<StateKey, float, StateHash> closed_cost;
    std::unordered_map<StateKey, Parent, StateHash> parents;

    std::size_t push_count = 0;
    std::shared_ptr<const puzzle::Environment> start_env = toShared(std::move(start));
    StateKey start_state = start_env->getState();
    best_cost.emplace(start_state, 0.0F);

    const std::vector<float> start_heuristics = cost_to_go.batch({start_env->costToGoInput()});
    if (start_heuristics.size() != 1) {
        throw std::runtime_error("batched heuristic returned wrong result count");
    }

    open.push(OpenEntry{
        priority(0.0F, start_heuristics[0], weight),
        push_count,
        0.0F,
        start_state,
        start_env,
    });

    while (!open.empty()) {
        std::vector<OpenEntry> popped;
        while (!open.empty() && static_cast<int>(popped.size()) < pop_batch_size) {
            OpenEntry entry = open.top();
            open.pop();

            const auto best_it = best_cost.find(entry.state);
            if (best_it == best_cost.end() || entry.path_cost != best_it->second) {
                continue;
            }

            const auto closed_it = closed_cost.find(entry.state);
            if (closed_it != closed_cost.end() && closed_it->second <= entry.path_cost) {
                continue;
            }

            if (entry.env->isSolved()) {
                return SearchResult{
                    true,
                    reconstructActions(entry.state, parents),
                    static_cast<int>(best_cost.size()),
                };
            }

            popped.push_back(std::move(entry));
        }

        if (popped.empty()) {
            continue;
        }

        std::vector<std::vector<float>> candidate_inputs;
        std::vector<Candidate> candidates;

        for (const OpenEntry& entry : popped) {
            closed_cost[entry.state] = entry.path_cost;

            for (int action : entry.env->getActions()) {
                std::unique_ptr<puzzle::Environment> child_unique = entry.env->getNextState(action);
                const float child_path_cost = entry.path_cost + 1.0F;
                StateKey child_state = child_unique->getState();

                const auto best_it = best_cost.find(child_state);
                const float previous_best = best_it == best_cost.end()
                    ? std::numeric_limits<float>::infinity()
                    : best_it->second;
                if (child_path_cost >= previous_best) {
                    continue;
                }
                if (best_it == best_cost.end() && static_cast<int>(best_cost.size()) >= max_states) {
                    continue;
                }

                std::shared_ptr<const puzzle::Environment> child_env = toShared(std::move(child_unique));
                best_cost[child_state] = child_path_cost;
                parents[child_state] = Parent{entry.state, action, 1.0F};
                candidate_inputs.push_back(child_env->costToGoInput());
                candidates.push_back(Candidate{child_path_cost, std::move(child_state), std::move(child_env)});
            }
        }

        if (candidates.empty()) {
            continue;
        }

        const std::vector<float> child_heuristics = cost_to_go.batch(candidate_inputs);
        if (child_heuristics.size() != candidates.size()) {
            throw std::runtime_error("batched heuristic returned wrong result count");
        }

        for (std::size_t i = 0; i < candidates.size(); ++i) {
            ++push_count;
            open.push(OpenEntry{
                priority(candidates[i].path_cost, child_heuristics[i], weight),
                push_count,
                candidates[i].path_cost,
                candidates[i].state,
                candidates[i].env,
            });
        }
    }

    return SearchResult{
        false,
        {},
        static_cast<int>(best_cost.size()),
    };
}

}  // namespace deepcube::search
