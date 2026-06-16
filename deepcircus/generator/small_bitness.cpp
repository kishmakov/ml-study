#include "small_bitness.h"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

struct SelectedBits {
    uint16_t mask = 0;
    uint16_t values = 0;

    bool operator==(const SelectedBits& other) const {
        return mask == other.mask && values == other.values;
    }
};

struct SelectedBitsHash {
    size_t operator()(SelectedBits bits) const {
        return static_cast<size_t>(bits.mask) | (static_cast<size_t>(bits.values) << 16);
    }
};

struct StateStats {
    uint8_t seen = 0;
};

struct StatePlan {
    bool is_leaf = true;
    bool value = false;
    uint16_t bit_id = 0;
    size_t cost = 0;
};

SelectedBits WithBit(SelectedBits bits, uint16_t bit_id, bool value) {
    const uint16_t bit = static_cast<uint16_t>(1u << bit_id);
    bits.mask |= bit;
    if (value) {
        bits.values |= bit;
    } else {
        bits.values &= static_cast<uint16_t>(~bit);
    }
    return bits;
}

class TableTreeBuilder {
public:
    TableTreeBuilder(uint16_t bitness, uint64_t truth_table)
        : bitness_(bitness)
        , truth_table_(truth_table)
        , tree_(bitness)
    {
        assert(IsSmallBitness(bitness_));
    }

    DecisionTree Build() {
        tree_.nodes.push_back(false);
        BuildNodes(0, {});
        return std::move(tree_);
    }

private:
    void BuildNodes(size_t node_id, SelectedBits bits) {
        const StatePlan plan = ChoosePlan(bits);

        if (plan.is_leaf) {
            tree_.nodes[node_id] = plan.value;
            ++tree_.num_leafs;
            return;
        }

        const size_t child0 = tree_.nodes.size();
        const size_t child1 = tree_.nodes.size() + 1;
        tree_.nodes[node_id] = Div{plan.bit_id, child0, child1};
        tree_.used_bits[plan.bit_id] = true;
        tree_.nodes.push_back(false);
        tree_.nodes.push_back(false);

        BuildNodes(child0, WithBit(bits, plan.bit_id, false));
        BuildNodes(child1, WithBit(bits, plan.bit_id, true));
    }

    StateStats Analyze(SelectedBits bits) {
        auto cached = stats_memo_.find(bits);
        if (cached != stats_memo_.end()) {
            return cached->second;
        }

        StateStats stats;
        const uint16_t all_bits = static_cast<uint16_t>((1u << bitness_) - 1);
        const uint16_t free_mask = static_cast<uint16_t>(all_bits ^ bits.mask);
        uint16_t sub = free_mask;
        do {
            const uint16_t input_id = static_cast<uint16_t>(sub | bits.values);
            stats.seen |= static_cast<uint8_t>(1u << TableValue(input_id));
            sub = static_cast<uint16_t>((sub - 1) & free_mask);
        } while (sub != free_mask);

        stats_memo_[bits] = stats;
        return stats;
    }

    StatePlan ChoosePlan(SelectedBits bits) {
        auto cached = plan_memo_.find(bits);
        if (cached != plan_memo_.end()) {
            return cached->second;
        }

        const StateStats stats = Analyze(bits);
        if (stats.seen != 3) {
            const StatePlan plan{true, stats.seen == 2, 0, 0};
            plan_memo_[bits] = plan;
            return plan;
        }

        StatePlan best;
        best.is_leaf = false;
        best.cost = std::numeric_limits<size_t>::max();

        for (uint16_t bit_id = 0; bit_id < bitness_; ++bit_id) {
            if ((bits.mask & (1u << bit_id)) != 0) {
                continue;
            }

            const StatePlan left = ChoosePlan(WithBit(bits, bit_id, false));
            const StatePlan right = ChoosePlan(WithBit(bits, bit_id, true));
            const size_t cost = 1 + left.cost + right.cost;
            if (cost < best.cost) {
                best.bit_id = bit_id;
                best.cost = cost;
            }
        }

        plan_memo_[bits] = best;
        return best;
    }

    bool TableValue(uint16_t input_id) const {
        return ((truth_table_ >> input_id) & 1ull) != 0;
    }

    uint16_t bitness_;
    uint64_t truth_table_;
    DecisionTree tree_;
    std::unordered_map<SelectedBits, StateStats, SelectedBitsHash> stats_memo_;
    std::unordered_map<SelectedBits, StatePlan, SelectedBitsHash> plan_memo_;
};

std::filesystem::path CacheDir(uint16_t bitness) {
    const std::filesystem::path cwd = std::filesystem::current_path();
    const std::filesystem::path tmp = cwd / "deepcircus" / "tmp";
    return tmp / ("b" + std::to_string(bitness));
}

std::filesystem::path CachePath(uint16_t bitness, uint64_t truth_table) {
    return CacheDir(bitness) / ("table_" + std::to_string(truth_table) + ".tree");
}

bool SaveTree(const DecisionTree& tree, const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        return false;
    }

    const uint32_t magic = 0x44544731; // DTG1
    const uint64_t node_count = tree.nodes.size();
    out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    out.write(reinterpret_cast<const char*>(&node_count), sizeof(node_count));
    for (const Node& node : tree.nodes) {
        const Div* division = std::get_if<Div>(&node);
        const uint8_t type = division == nullptr ? 0 : 1;
        out.write(reinterpret_cast<const char*>(&type), sizeof(type));
        if (division == nullptr) {
            const uint8_t value = std::get<bool>(node) ? 1 : 0;
            out.write(reinterpret_cast<const char*>(&value), sizeof(value));
        } else {
            const uint16_t bit_id = static_cast<uint16_t>(division->bitId);
            const uint64_t child0 = division->child0;
            const uint64_t child1 = division->child1;
            out.write(reinterpret_cast<const char*>(&bit_id), sizeof(bit_id));
            out.write(reinterpret_cast<const char*>(&child0), sizeof(child0));
            out.write(reinterpret_cast<const char*>(&child1), sizeof(child1));
        }
    }

    return out.good();
}

bool LoadTree(uint16_t bitness, const std::filesystem::path& path, DecisionTree& tree) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }

    uint32_t magic = 0;
    uint64_t node_count = 0;
    in.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    in.read(reinterpret_cast<char*>(&node_count), sizeof(node_count));
    if (!in || magic != 0x44544731) {
        return false;
    }

    DecisionTree loaded(bitness);
    loaded.nodes.reserve(static_cast<size_t>(node_count));

    for (uint64_t node_id = 0; node_id < node_count; ++node_id) {
        uint8_t type = 0;
        in.read(reinterpret_cast<char*>(&type), sizeof(type));
        if (!in) {
            return false;
        }

        if (type == 0) {
            uint8_t value = 0;
            in.read(reinterpret_cast<char*>(&value), sizeof(value));
            if (!in) {
                return false;
            }
            loaded.nodes.push_back(value != 0);
            ++loaded.num_leafs;
        } else if (type == 1) {
            uint16_t bit_id = 0;
            uint64_t child0 = 0;
            uint64_t child1 = 0;
            in.read(reinterpret_cast<char*>(&bit_id), sizeof(bit_id));
            in.read(reinterpret_cast<char*>(&child0), sizeof(child0));
            in.read(reinterpret_cast<char*>(&child1), sizeof(child1));
            if (!in || bit_id >= bitness || child0 >= node_count || child1 >= node_count) {
                return false;
            }
            loaded.nodes.push_back(Div{bit_id, static_cast<size_t>(child0), static_cast<size_t>(child1)});
            loaded.used_bits[bit_id] = true;
        } else {
            return false;
        }
    }

    tree = std::move(loaded);
    return true;
}

bool IsSmallBitness(uint16_t bitness) {
    return bitness <= kExactTableBitness;
}

size_t SmallBitnessCasesNumber(uint16_t bitness) {
    assert(IsSmallBitness(bitness));
    return 1ull << (1ull << bitness);
}

DecisionTree SmallBitnessTree(uint16_t bitness, size_t case_id) {
    assert(IsSmallBitness(bitness));
    assert(case_id < SmallBitnessCasesNumber(bitness));

    DecisionTree tree(bitness);
    const std::filesystem::path path = CachePath(bitness, case_id);
    if (LoadTree(bitness, path, tree)) {
        return tree;
    }

    tree = TableTreeBuilder(bitness, case_id).Build();
    (void)SaveTree(tree, path);
    return tree;
}
