#include "search/a_star.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
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

constexpr std::uint32_t kInvalidStateId = std::numeric_limits<std::uint32_t>::max();

struct InternResult {
    std::uint32_t id = kInvalidStateId;
    bool inserted = false;
};

struct StateInterner {
    std::unordered_map<StateKey, std::uint32_t, StateHash> ids;
    std::vector<StateKey> states;

    void reserve(std::size_t count) {
        ids.reserve(count * 2U);
        states.reserve(count);
    }

    InternResult intern(StateKey key, std::size_t max_states) {
        const auto existing = ids.find(key);
        if (existing != ids.end()) {
            return InternResult{existing->second, false};
        }
        if (states.size() >= max_states) {
            return InternResult{kInvalidStateId, false};
        }

        const std::uint32_t id = static_cast<std::uint32_t>(states.size());
        auto [it, inserted] = ids.emplace(std::move(key), id);
        if (!inserted) {
            return InternResult{it->second, false};
        }

        states.push_back(it->first);
        return InternResult{id, true};
    }

    std::size_t size() const {
        return states.size();
    }
};

enum class NodeStatus : std::uint8_t {
    Open,
    Closed,
};

struct NodeInfo {
    float cost = std::numeric_limits<float>::infinity();
    NodeStatus status = NodeStatus::Open;
};

struct Parent {
    std::uint32_t state_id = kInvalidStateId;
    std::int8_t action = -1;
};

struct OpenEntry {
    float priority = 0.0F;
    std::size_t push_count = 0;
    float path_cost = 0.0F;
    std::uint32_t state_id = kInvalidStateId;
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
    std::uint32_t state_id = kInvalidStateId;
};

float priority(float path_cost, float heuristic, float weight) {
    return weight * path_cost + heuristic;
}

std::vector<int> reconstructActions(
    std::uint32_t goal_id,
    const std::vector<Parent>& parents
) {
    std::vector<int> actions;
    std::uint32_t current = goal_id;

    while (true) {
        const Parent& parent = parents[current];
        if (parent.state_id == kInvalidStateId) {
            break;
        }

        actions.push_back(static_cast<int>(parent.action));
        current = parent.state_id;
    }

    std::reverse(actions.begin(), actions.end());
    return actions;
}

}  // namespace

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

    const std::size_t max_state_count = static_cast<std::size_t>(max_states);
    std::priority_queue<OpenEntry, std::vector<OpenEntry>, OpenEntryCompare> open;
    StateInterner interner;
    interner.reserve(max_state_count);
    std::vector<NodeInfo> node_info;
    std::vector<Parent> parents;
    std::vector<std::unique_ptr<puzzle::Environment>> envs;
    node_info.reserve(max_state_count);
    parents.reserve(max_state_count);
    envs.reserve(max_state_count);

    std::size_t push_count = 0;
    StateKey start_state = start->getState();
    const InternResult start_intern = interner.intern(std::move(start_state), max_state_count);
    if (!start_intern.inserted) {
        throw std::runtime_error("failed to intern A* start state");
    }
    node_info.push_back(NodeInfo{0.0F, NodeStatus::Open});
    parents.push_back(Parent{});
    envs.push_back(std::move(start));

    const std::vector<float> start_heuristics = cost_to_go.batch({envs[start_intern.id]->costToGoInput()});
    if (start_heuristics.size() != 1) {
        throw std::runtime_error("batched heuristic returned wrong result count");
    }

    open.push(OpenEntry{
        priority(0.0F, start_heuristics[0], weight),
        push_count,
        0.0F,
        start_intern.id,
    });

    std::vector<int> actions;
    std::vector<OpenEntry> popped;
    std::vector<std::vector<float>> candidate_inputs;
    std::vector<Candidate> candidates;
    popped.reserve(static_cast<std::size_t>(pop_batch_size));
    candidate_inputs.reserve(static_cast<std::size_t>(pop_batch_size) * 12U);
    candidates.reserve(static_cast<std::size_t>(pop_batch_size) * 12U);

    while (!open.empty()) {
        popped.clear();
        while (!open.empty() && static_cast<int>(popped.size()) < pop_batch_size) {
            OpenEntry entry = open.top();
            open.pop();

            const NodeInfo& info = node_info[entry.state_id];
            if (info.status == NodeStatus::Closed || entry.path_cost > info.cost) {
                continue;
            }

            if (envs[entry.state_id]->isSolved()) {
                return SearchResult{
                    true,
                    reconstructActions(entry.state_id, parents),
                    static_cast<int>(interner.size()),
                };
            }

            popped.push_back(std::move(entry));
        }

        if (popped.empty()) {
            continue;
        }

        candidate_inputs.clear();
        candidates.clear();

        for (const OpenEntry& entry : popped) {
            node_info[entry.state_id].status = NodeStatus::Closed;
            const std::unique_ptr<puzzle::Environment>& env = envs[entry.state_id];

            env->getActions(actions);
            for (int action : actions) {
                std::unique_ptr<puzzle::Environment> child_unique = env->getNextState(action);
                const float child_path_cost = entry.path_cost + 1.0F;
                StateKey child_state = child_unique->getState();

                const InternResult child_intern = interner.intern(std::move(child_state), max_state_count);
                if (child_intern.id == kInvalidStateId) {
                    continue;
                }

                if (child_intern.inserted) {
                    node_info.push_back(NodeInfo{child_path_cost, NodeStatus::Open});
                    parents.push_back(Parent{
                        entry.state_id,
                        static_cast<std::int8_t>(action),
                    });
                    envs.push_back(std::move(child_unique));
                } else {
                    NodeInfo& child_info = node_info[child_intern.id];
                    if (child_info.status == NodeStatus::Closed || child_path_cost >= child_info.cost) {
                        continue;
                    }
                    child_info.cost = child_path_cost;
                    parents[child_intern.id] = Parent{
                        entry.state_id,
                        static_cast<std::int8_t>(action),
                    };
                }

                candidate_inputs.push_back(envs[child_intern.id]->costToGoInput());
                candidates.push_back(Candidate{child_path_cost, child_intern.id});
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
                candidates[i].state_id,
            });
        }
    }

    return SearchResult{
        false,
        {},
        static_cast<int>(interner.size()),
    };
}

}  // namespace deepcube::search
