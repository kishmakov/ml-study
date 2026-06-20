#include "small_bitness.h"

#include <cassert>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

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
        tree_.Finalize();
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

std::filesystem::path CacheDir() {
    const std::filesystem::path cwd = std::filesystem::current_path();
    return cwd / "tmp";
}

std::filesystem::path CachePath() {
    return CacheDir() / "small_trees.treepack";
}

uint64_t StreamOffset(std::streampos pos) {
    return static_cast<uint64_t>(static_cast<std::streamoff>(pos));
}

class MappedFile {
public:
    explicit MappedFile(const std::filesystem::path& path) {
        const std::string path_string = path.string();
        fd_ = open(path_string.c_str(), O_RDONLY);
        if (fd_ < 0) {
            return;
        }

        struct stat status;
        if (fstat(fd_, &status) != 0 || status.st_size <= 0) {
            return;
        }

        size_ = static_cast<size_t>(status.st_size);
        data_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            size_ = 0;
        }
    }

    ~MappedFile() {
        if (data_ != nullptr) {
            munmap(data_, size_);
        }
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    explicit operator bool() const {
        return data_ != nullptr;
    }

    const uint8_t* Data() const {
        return static_cast<const uint8_t*>(data_);
    }

    size_t Size() const {
        return size_;
    }

private:
    int fd_ = -1;
    void* data_ = nullptr;
    size_t size_ = 0;
};

template <class T>
bool ReadPod(const uint8_t*& cursor, const uint8_t* end, T& value) {
    if (cursor > end || static_cast<size_t>(end - cursor) < sizeof(T)) {
        return false;
    }

    std::memcpy(&value, cursor, sizeof(T));
    cursor += sizeof(T);
    return true;
}

bool WriteTree(std::ostream& out, const DecisionTree& tree) {
    const uint64_t node_count = tree.nodes.size();
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

bool ReadTree(const uint8_t*& cursor, const uint8_t* end, uint16_t bitness, DecisionTree& tree) {
    uint64_t node_count = 0;
    if (!ReadPod(cursor, end, node_count) ||
        node_count == 0 ||
        node_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        return false;
    }

    DecisionTree loaded(bitness);
    loaded.nodes.reserve(static_cast<size_t>(node_count));

    for (uint64_t node_id = 0; node_id < node_count; ++node_id) {
        uint8_t type = 0;
        if (!ReadPod(cursor, end, type)) {
            return false;
        }

        if (type == 0) {
            uint8_t value = 0;
            if (!ReadPod(cursor, end, value)) {
                return false;
            }
            loaded.nodes.push_back(value != 0);
            ++loaded.num_leafs;
        } else if (type == 1) {
            uint16_t bit_id = 0;
            uint64_t child0 = 0;
            uint64_t child1 = 0;
            if (!ReadPod(cursor, end, bit_id) ||
                !ReadPod(cursor, end, child0) ||
                !ReadPod(cursor, end, child1) ||
                bit_id >= bitness || child0 >= node_count || child1 >= node_count) {
                return false;
            }
            loaded.nodes.push_back(Div{bit_id, static_cast<size_t>(child0), static_cast<size_t>(child1)});
            loaded.used_bits[bit_id] = true;
        } else {
            return false;
        }
    }

    loaded.Finalize();
    tree = std::move(loaded);
    return true;
}

bool SaveTrees(const std::filesystem::path& path) {
    std::error_code error;
    std::filesystem::create_directories(path.parent_path(), error);
    if (error) {
        return false;
    }

    const uint32_t magic = 0x44544734; // DTG4
    const uint16_t max_bitness = kExactTableBitness;

    const std::filesystem::path tmp_path = path.parent_path() / (path.filename().string() + ".tmp");
    {
        std::ofstream out(tmp_path, std::ios::binary);
        if (!out) {
            return false;
        }

        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&max_bitness), sizeof(max_bitness));
        for (uint16_t bitness = 0; bitness <= kExactTableBitness; ++bitness) {
            const uint64_t cases_number = SmallBitnessCasesNumber(bitness);
            const uint64_t empty_offset = 0;
            out.write(reinterpret_cast<const char*>(&bitness), sizeof(bitness));
            out.write(reinterpret_cast<const char*>(&cases_number), sizeof(cases_number));
            const std::streampos group_end_pos = out.tellp();
            out.write(reinterpret_cast<const char*>(&empty_offset), sizeof(empty_offset));

            const std::streampos offsets_pos = out.tellp();
            for (uint64_t case_id = 0; case_id < cases_number; ++case_id) {
                out.write(reinterpret_cast<const char*>(&empty_offset), sizeof(empty_offset));
            }

            std::vector<uint64_t> offsets;
            offsets.reserve(static_cast<size_t>(cases_number));
            for (uint64_t case_id = 0; case_id < cases_number; ++case_id) {
                offsets.push_back(StreamOffset(out.tellp()));
                const DecisionTree tree = TableTreeBuilder(bitness, case_id).Build();
                if (!WriteTree(out, tree)) {
                    return false;
                }
            }

            const uint64_t group_end = StreamOffset(out.tellp());
            out.seekp(group_end_pos);
            out.write(reinterpret_cast<const char*>(&group_end), sizeof(group_end));
            out.seekp(offsets_pos);
            for (uint64_t offset : offsets) {
                out.write(reinterpret_cast<const char*>(&offset), sizeof(offset));
            }
            out.seekp(static_cast<std::streamoff>(group_end));
        }

        if (!out.good()) {
            return false;
        }
    }

    std::filesystem::rename(tmp_path, path, error);
    return !error;
}

bool LoadTree(uint16_t bitness, size_t case_id, const std::filesystem::path& path, DecisionTree& tree) {
    const MappedFile file(path);
    if (!file) {
        return false;
    }

    const uint8_t* const begin = file.Data();
    const uint8_t* const end = begin + file.Size();
    const uint8_t* cursor = begin;

    uint32_t magic = 0;
    uint16_t max_bitness = 0;
    if (!ReadPod(cursor, end, magic) ||
        !ReadPod(cursor, end, max_bitness) ||
        magic != 0x44544734 || max_bitness != kExactTableBitness) {
        return false;
    }

    for (uint16_t group_bitness = 0; group_bitness <= kExactTableBitness; ++group_bitness) {
        uint16_t stored_bitness = 0;
        uint64_t cases_number = 0;
        uint64_t group_end = 0;
        if (!ReadPod(cursor, end, stored_bitness) ||
            !ReadPod(cursor, end, cases_number) ||
            !ReadPod(cursor, end, group_end) ||
            stored_bitness != group_bitness ||
            cases_number != SmallBitnessCasesNumber(group_bitness) ||
            group_end < static_cast<uint64_t>(cursor - begin) ||
            group_end > file.Size()) {
            return false;
        }

        if (stored_bitness != bitness) {
            cursor = begin + group_end;
            continue;
        }

        if (case_id >= cases_number) {
            return false;
        }
        const uint64_t offsets_pos = static_cast<uint64_t>(cursor - begin);
        const uint64_t offset_pos = offsets_pos + static_cast<uint64_t>(case_id) * sizeof(uint64_t);
        if (offset_pos + sizeof(uint64_t) > group_end) {
            return false;
        }

        uint64_t tree_offset = 0;
        const uint8_t* offset_cursor = begin + offset_pos;
        if (!ReadPod(offset_cursor, end, tree_offset) || tree_offset >= group_end) {
            return false;
        }

        const uint8_t* tree_cursor = begin + tree_offset;
        return ReadTree(tree_cursor, begin + group_end, bitness, tree);
    }

    return false;
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
    const std::filesystem::path path = CachePath();
    if (LoadTree(bitness, case_id, path, tree)) {
        return tree;
    }

    (void)SaveTrees(path);
    if (LoadTree(bitness, case_id, path, tree)) {
        return tree;
    }

    tree = TableTreeBuilder(bitness, case_id).Build();
    return tree;
}
