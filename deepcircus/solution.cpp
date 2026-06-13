#include "solution.h"

#include <algorithm>
#include <cassert>
#include <limits>
#include <unordered_map>
#include <vector>

namespace {

constexpr uint64_t kExactFreeBitsLimit = 8;

struct SelectedBits
{
    uint64_t mask;
    uint64_t values;

    bool operator==(const SelectedBits& other) const {
        return mask == other.mask && values == other.values;
    }

    [[nodiscard]] uint64_t FreeNum(uint64_t N) const {
        return __builtin_popcountll(MaskComplement(N));
    }

    [[nodiscard]] uint64_t MaskComplement(uint64_t N) const {
        return ((1ull << N) - 1) ^ mask;
    }
};

struct SelectedBitsHash
{
    size_t operator()(SelectedBits bits) const {
        const uint64_t mixed = bits.mask ^ (bits.values + 0x9e3779b97f4a7c15ull
            + (bits.mask << 6) + (bits.mask >> 2));
        return static_cast<size_t>(mixed);
    }
};

struct StateStats
{
    int seen = 0;
    size_t ones = 0;
    size_t total = 0;
};

struct StatePlan
{
    bool isLeaf = true;
    bool value = false;
    uint64_t bitId = 0;
    size_t cost = 0;
};

SelectedBits WithBit(SelectedBits bits, uint64_t bitId, bool value) {
    const uint64_t bit = 1ull << bitId;
    bits.mask |= bit;
    if (value) {
        bits.values |= bit;
    } else {
        bits.values &= ~bit;
    }
    return bits;
}

class TreeBuilder
{
public:
    TreeBuilder(uint64_t N, const std::function<bool(const std::string&)>& func)
        : N_(N)
        , func_(func)
    {
        assert(N_ < 63);
    }

    void BuildNodes(size_t currentId, SelectedBits bits, std::vector<Node>& nodes) {
        const StatePlan plan = ChoosePlan(bits);

        if (plan.isLeaf) {
            nodes[currentId].division = std::nullopt;
            nodes[currentId].value = plan.value;
            return;
        }

        const uint64_t l0 = nodes.size();
        const uint64_t l1 = nodes.size() + 1;
        nodes[currentId].division = {plan.bitId, l0, l1};
        nodes.emplace_back();
        nodes.emplace_back();

        BuildNodes(l0, WithBit(bits, plan.bitId, false), nodes);
        BuildNodes(l1, WithBit(bits, plan.bitId, true), nodes);
    }

private:
    StateStats Analyze(SelectedBits bits) {
        auto cached = statsMemo_.find(bits);
        if (cached != statsMemo_.end()) {
            return cached->second;
        }

        StateStats stats;
        const uint64_t maskComplement = bits.MaskComplement(N_);
        uint64_t sub = maskComplement;
        do {
            const uint64_t id = sub | bits.values;
            const uint64_t value = Value(id);
            stats.seen |= 1 << value;
            stats.ones += value;
            ++stats.total;
            sub = (sub - 1) & maskComplement;
        } while (sub != maskComplement);

        assert(stats.seen > 0);
        statsMemo_[bits] = stats;
        return stats;
    }

    StatePlan LeafPlan(const StateStats& stats) const {
        return {true, stats.seen == 2, 0, 0};
    }

    StatePlan ChoosePlan(SelectedBits bits) {
        const StateStats stats = Analyze(bits);
        if (stats.seen != 3) return LeafPlan(stats);
        if (bits.FreeNum(N_) <= kExactFreeBitsLimit) return ExactPlan(bits);
        return GreedyPlan(bits);
    }

    StatePlan ExactPlan(SelectedBits bits) {
        auto cached = exactMemo_.find(bits);
        if (cached != exactMemo_.end()) {
            return cached->second;
        }

        const StateStats stats = Analyze(bits);
        if (stats.seen != 3) {
            const StatePlan plan = LeafPlan(stats);
            exactMemo_[bits] = plan;
            return plan;
        }

        StatePlan best;
        best.isLeaf = false;
        best.cost = std::numeric_limits<size_t>::max();

        for (uint64_t bitId = 0; bitId < N_; ++bitId) {
            if ((bits.mask & (1ull << bitId)) != 0) {
                continue;
            }

            const StatePlan left = ExactPlan(WithBit(bits, bitId, false));
            const StatePlan right = ExactPlan(WithBit(bits, bitId, true));
            const size_t cost = 1 + left.cost + right.cost;
            if (cost < best.cost) {
                best.bitId = bitId;
                best.cost = cost;
            }
        }

        exactMemo_[bits] = best;
        return best;
    }

    StatePlan GreedyPlan(SelectedBits bits) {
        StatePlan best;
        best.isLeaf = false;

        size_t bestScore = std::numeric_limits<size_t>::max();
        size_t bestMixedChildren = std::numeric_limits<size_t>::max();
        size_t bestWorstChildScore = std::numeric_limits<size_t>::max();
        size_t bestExactCost = std::numeric_limits<size_t>::max();

        for (uint64_t bitId = 0; bitId < N_; ++bitId) {
            if ((bits.mask & (1ull << bitId)) != 0) {
                continue;
            }

            const SelectedBits leftBits = WithBit(bits, bitId, false);
            const SelectedBits rightBits = WithBit(bits, bitId, true);
            const size_t leftScore = ChildScore(leftBits);
            const size_t rightScore = ChildScore(rightBits);
            const size_t score = leftScore + rightScore;
            const size_t mixedChildren = (leftScore != 0 ? 1 : 0) + (rightScore != 0 ? 1 : 0);
            const size_t worstChildScore = std::max(leftScore, rightScore);

            size_t exactCost = std::numeric_limits<size_t>::max();
            if (leftBits.FreeNum(N_) <= kExactFreeBitsLimit) {
                exactCost = 1 + ExactPlan(leftBits).cost + ExactPlan(rightBits).cost;
            }

            if (score < bestScore
                || (score == bestScore && mixedChildren < bestMixedChildren)
                || (score == bestScore && mixedChildren == bestMixedChildren
                    && worstChildScore < bestWorstChildScore)
                || (score == bestScore && mixedChildren == bestMixedChildren
                    && worstChildScore == bestWorstChildScore && exactCost < bestExactCost)) {
                best.bitId = bitId;
                bestScore = score;
                bestMixedChildren = mixedChildren;
                bestWorstChildScore = worstChildScore;
                bestExactCost = exactCost;
                best.cost = exactCost;
            }
        }

        assert(bestScore != std::numeric_limits<size_t>::max());
        return best;
    }

    size_t ChildScore(SelectedBits bits) {
        const StateStats stats = Analyze(bits);
        if (stats.seen != 3) {
            return 0;
        }

        if (bits.FreeNum(N_) <= kExactFreeBitsLimit) {
            return ExactPlan(bits).cost;
        }

        return std::min(stats.ones, stats.total - stats.ones);
    }

    uint64_t N_;
    const std::function<bool(const std::string&)>& func_;
    std::unordered_map<uint64_t, uint64_t> valueMemo_;
    std::unordered_map<SelectedBits, StateStats, SelectedBitsHash> statsMemo_;
    std::unordered_map<SelectedBits, StatePlan, SelectedBitsHash> exactMemo_;

    uint64_t Value(uint64_t id) {
        auto cached = valueMemo_.find(id);
        if (cached != valueMemo_.end()) {
            return cached->second;
        }

        std::string input(N_, '0');
        for (uint64_t bit = 0; bit < N_; ++bit) {
            input[bit] = ((id >> bit) & 1ull) != 0 ? '1' : '0';
        }

        const uint64_t value = func_(input) ? 1 : 0;
        valueMemo_[id] = value;
        return value;
    }
};

}  // namespace

std::vector<Node> Solve(uint64_t N, const std::function<bool(const std::string&)>& func) {
    std::vector<Node> nodes(0);
    nodes.emplace_back(std::nullopt, false);
    TreeBuilder builder(N, func);
    builder.BuildNodes(0, {0, 0}, nodes);
    return nodes;
}
