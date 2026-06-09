#include "environment.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace deepcube::puzzle {
namespace {

std::vector<std::string> split(const std::string& raw, char delimiter, std::size_t max_splits) {
    std::vector<std::string> parts;
    std::size_t start = 0;

    while (parts.size() < max_splits) {
        const std::size_t end = raw.find(delimiter, start);
        if (end == std::string::npos) {
            break;
        }
        parts.push_back(raw.substr(start, end - start));
        start = end + 1;
    }

    parts.push_back(raw.substr(start));
    return parts;
}

std::vector<int> parse_int_list(const std::string& raw, char delimiter) {
    std::vector<int> values;
    std::size_t start = 0;
    while (start <= raw.size()) {
        const std::size_t end = raw.find(delimiter, start);
        const std::string token = raw.substr(
            start,
            end == std::string::npos ? std::string::npos : end - start
        );
        if (token.empty()) {
            throw std::runtime_error("empty integer token in state: " + raw);
        }
        values.push_back(std::stoi(token));
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    return values;
}

std::vector<uint8_t> to_u8_vector(const std::vector<int>& values) {
    std::vector<uint8_t> output;
    output.reserve(values.size());
    for (int value : values) {
        if (value < 0 || value > 255) {
            throw std::runtime_error("state value is outside uint8 range");
        }
        output.push_back(static_cast<uint8_t>(value));
    }
    return output;
}

void validate_permutation(const std::vector<uint8_t>& values, int size, const std::string& name) {
    if (static_cast<int>(values.size()) != size) {
        throw std::runtime_error(name + " has wrong length");
    }

    std::vector<int> sorted(values.begin(), values.end());
    std::sort(sorted.begin(), sorted.end());
    for (int i = 0; i < size; ++i) {
        if (sorted[i] != i) {
            throw std::runtime_error(name + " is not a permutation");
        }
    }
}

void validate_bounded(const std::vector<uint8_t>& values, int size, int bound, const std::string& name) {
    if (static_cast<int>(values.size()) != size) {
        throw std::runtime_error(name + " has wrong length");
    }
    for (uint8_t value : values) {
        if (value >= bound) {
            throw std::runtime_error(name + " contains out-of-range orientation");
        }
    }
}

class NPuzzle : public Environment {
public:
    NPuzzle(std::vector<uint8_t> state, uint8_t dim)
        : state_(std::move(state)),
          dim_(dim),
          num_tiles_(dim_ * dim_) {
        validate();
        const auto zero = std::find(state_.begin(), state_.end(), 0);
        z_idx_ = static_cast<int>(zero - state_.begin());
    }

    static NPuzzle fromKey(const std::string& state_key, uint8_t dim) {
        return NPuzzle(to_u8_vector(parse_int_list(state_key, ',')), dim);
    }

    std::unique_ptr<Environment> getNextState(int action) const override {
        return std::make_unique<NPuzzle>(nextState(action));
    }

    std::vector<std::unique_ptr<Environment>> getNextStates() const override {
        std::vector<std::unique_ptr<Environment>> states;
        for (int action : actions()) {
            states.push_back(getNextState(action));
        }
        return states;
    }

    std::vector<uint8_t> getState() const override {
        return state_;
    }

    std::vector<float> costToGoInput() const override {
        std::vector<float> input;
        input.reserve(state_.size());
        for (uint8_t tile : state_) {
            input.push_back(static_cast<float>(tile));
        }
        return input;
    }

    bool isSolved() const override {
        for (int i = 0; i < num_tiles_ - 1; ++i) {
            if (state_[i] != i + 1) {
                return false;
            }
        }
        return state_.back() == 0;
    }

    int getNumActions() const override {
        return 4;
    }

private:
    std::vector<int> actions() const {
        std::vector<int> output;
        const int row = z_idx_ / dim_;
        const int col = z_idx_ % dim_;
        if (row < dim_ - 1) {
            output.push_back(0);
        }
        if (row > 0) {
            output.push_back(1);
        }
        if (col < dim_ - 1) {
            output.push_back(2);
        }
        if (col > 0) {
            output.push_back(3);
        }
        return output;
    }

    NPuzzle nextState(int action) const {
        int swap_idx = z_idx_;
        const int row = z_idx_ / dim_;
        const int col = z_idx_ % dim_;

        if (action == 0) {
            if (row >= dim_ - 1) {
                throw std::runtime_error("Cannot apply U at bottom row");
            }
            swap_idx = z_idx_ + dim_;
        } else if (action == 1) {
            if (row <= 0) {
                throw std::runtime_error("Cannot apply D at top row");
            }
            swap_idx = z_idx_ - dim_;
        } else if (action == 2) {
            if (col >= dim_ - 1) {
                throw std::runtime_error("Cannot apply L at right edge");
            }
            swap_idx = z_idx_ + 1;
        } else if (action == 3) {
            if (col <= 0) {
                throw std::runtime_error("Cannot apply R at left edge");
            }
            swap_idx = z_idx_ - 1;
        } else {
            throw std::runtime_error("Invalid NPuzzle action");
        }

        std::vector<uint8_t> next = state_;
        std::swap(next[z_idx_], next[swap_idx]);
        return NPuzzle(std::move(next), dim_);
    }

    void validate() const {
        if (dim_ < 2) {
            throw std::runtime_error("NPuzzle dimension must be at least 2");
        }
        validate_permutation(state_, num_tiles_, "NPuzzle state");
    }

    std::vector<uint8_t> state_;
    uint8_t dim_;
    int num_tiles_;
    int z_idx_ = 0;
};

struct Cube3State {
    std::array<uint8_t, 8> corner_perm{};
    std::array<uint8_t, 8> corner_ori{};
    std::array<uint8_t, 12> edge_perm{};
    std::array<uint8_t, 12> edge_ori{};

    bool operator==(const Cube3State& other) const {
        return corner_perm == other.corner_perm
            && corner_ori == other.corner_ori
            && edge_perm == other.edge_perm
            && edge_ori == other.edge_ori;
    }
};

template <std::size_t Size>
std::array<uint8_t, Size> to_u8_array(const std::vector<int>& values, const std::string& name) {
    if (values.size() != Size) {
        throw std::runtime_error(name + " has wrong length");
    }

    std::array<uint8_t, Size> output{};
    for (std::size_t i = 0; i < Size; ++i) {
        if (values[i] < 0 || values[i] > 255) {
            throw std::runtime_error(name + " contains out-of-range value");
        }
        output[i] = static_cast<uint8_t>(values[i]);
    }
    return output;
}

template <std::size_t Size>
std::vector<uint8_t> to_vector(const std::array<uint8_t, Size>& values) {
    return std::vector<uint8_t>(values.begin(), values.end());
}

class Cube3 : public Environment {
public:
    explicit Cube3(Cube3State state)
        : state_(std::move(state)) {
        validate();
    }

    static Cube3 fromKey(const std::string& state_key) {
        std::vector<std::string> parts = split(state_key, '|', 3);
        if (parts.size() != 4) {
            throw std::runtime_error("Cube3 state key must have four cubie fields");
        }

        return Cube3(Cube3State{
            to_u8_array<8>(parse_int_list(parts[0], ','), "Cube3 corner_perm"),
            to_u8_array<8>(parse_int_list(parts[1], ','), "Cube3 corner_ori"),
            to_u8_array<12>(parse_int_list(parts[2], ','), "Cube3 edge_perm"),
            to_u8_array<12>(parse_int_list(parts[3], ','), "Cube3 edge_ori"),
        });
    }

    std::unique_ptr<Environment> getNextState(int action) const override {
        return std::make_unique<Cube3>(applyAction(action));
    }

    std::vector<std::unique_ptr<Environment>> getNextStates() const override {
        std::vector<std::unique_ptr<Environment>> states;
        states.reserve(kNumActions);
        for (int action = 0; action < kNumActions; ++action) {
            states.push_back(getNextState(action));
        }
        return states;
    }

    std::vector<uint8_t> getState() const override {
        std::vector<uint8_t> state;
        state.reserve(40);
        state.insert(state.end(), state_.corner_perm.begin(), state_.corner_perm.end());
        state.insert(state.end(), state_.corner_ori.begin(), state_.corner_ori.end());
        state.insert(state.end(), state_.edge_perm.begin(), state_.edge_perm.end());
        state.insert(state.end(), state_.edge_ori.begin(), state_.edge_ori.end());
        return state;
    }

    std::vector<float> costToGoInput() const override {
        const std::array<uint8_t, 54> colors = faceColors();
        std::vector<float> input(54 * 6, 0.0F);
        for (std::size_t i = 0; i < colors.size(); ++i) {
            input[i * 6 + colors[i]] = 1.0F;
        }
        return input;
    }

    bool isSolved() const override {
        const std::array<uint8_t, 54> colors = faceColors();
        for (int face = 0; face < 6; ++face) {
            const uint8_t color = colors[face * 9];
            for (int offset = 1; offset < 9; ++offset) {
                if (colors[face * 9 + offset] != color) {
                    return false;
                }
            }
        }
        return true;
    }

    int getNumActions() const override {
        return kNumActions;
    }

private:
    static constexpr int kNumActions = 12;

    static constexpr std::array<std::array<uint8_t, 3>, 8> kCornerFacelets{{
        {{0, 26, 47}}, {{2, 44, 20}}, {{6, 53, 29}}, {{8, 35, 38}},
        {{9, 18, 42}}, {{11, 45, 24}}, {{15, 36, 33}}, {{17, 27, 51}},
    }};

    static constexpr std::array<std::array<uint8_t, 2>, 12> kEdgeFacelets{{
        {{1, 23}}, {{3, 50}}, {{5, 41}}, {{7, 32}},
        {{10, 21}}, {{12, 39}}, {{14, 48}}, {{16, 30}},
        {{19, 43}}, {{25, 46}}, {{28, 52}}, {{34, 37}},
    }};

    static constexpr std::array<std::array<uint8_t, 8>, kNumActions> kActionCornerPerm{{
        {{1, 3, 0, 2, 4, 5, 6, 7}},
        {{2, 0, 3, 1, 4, 5, 6, 7}},
        {{0, 1, 2, 3, 5, 7, 4, 6}},
        {{0, 1, 2, 3, 6, 4, 7, 5}},
        {{5, 0, 2, 3, 1, 4, 6, 7}},
        {{1, 4, 2, 3, 5, 0, 6, 7}},
        {{0, 1, 3, 6, 4, 5, 7, 2}},
        {{0, 1, 7, 2, 4, 5, 3, 6}},
        {{0, 4, 2, 1, 6, 5, 3, 7}},
        {{0, 3, 2, 6, 1, 5, 4, 7}},
        {{2, 1, 7, 3, 4, 0, 6, 5}},
        {{5, 1, 0, 3, 4, 7, 6, 2}},
    }};

    static constexpr std::array<std::array<uint8_t, 8>, kNumActions> kActionCornerOri{{
        {{0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0}},
        {{1, 2, 0, 0, 1, 2, 0, 0}},
        {{1, 2, 0, 0, 1, 2, 0, 0}},
        {{0, 0, 2, 1, 0, 0, 2, 1}},
        {{0, 0, 2, 1, 0, 0, 2, 1}},
        {{0, 1, 0, 2, 2, 0, 1, 0}},
        {{0, 1, 0, 2, 2, 0, 1, 0}},
        {{2, 0, 1, 0, 0, 1, 0, 2}},
        {{2, 0, 1, 0, 0, 1, 0, 2}},
    }};

    static constexpr std::array<std::array<uint8_t, 12>, kNumActions> kActionEdgePerm{{
        {{2, 0, 3, 1, 4, 5, 6, 7, 8, 9, 10, 11}},
        {{1, 3, 0, 2, 4, 5, 6, 7, 8, 9, 10, 11}},
        {{0, 1, 2, 3, 6, 4, 7, 5, 8, 9, 10, 11}},
        {{0, 1, 2, 3, 5, 7, 4, 6, 8, 9, 10, 11}},
        {{9, 1, 2, 3, 8, 5, 6, 7, 0, 4, 10, 11}},
        {{8, 1, 2, 3, 9, 5, 6, 7, 4, 0, 10, 11}},
        {{0, 1, 2, 11, 4, 5, 6, 10, 8, 9, 3, 7}},
        {{0, 1, 2, 10, 4, 5, 6, 11, 8, 9, 7, 3}},
        {{0, 1, 8, 3, 4, 11, 6, 7, 5, 9, 10, 2}},
        {{0, 1, 11, 3, 4, 8, 6, 7, 2, 9, 10, 5}},
        {{0, 10, 2, 3, 4, 5, 9, 7, 8, 1, 6, 11}},
        {{0, 9, 2, 3, 4, 5, 10, 7, 8, 6, 1, 11}},
    }};

    static constexpr std::array<std::array<uint8_t, 12>, kNumActions> kActionEdgeOri{{
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0}},
        {{1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0}},
        {{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1}},
        {{0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1}},
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
        {{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
    }};

    Cube3 applyAction(int action) const {
        if (action < 0 || action >= kNumActions) {
            throw std::runtime_error("Invalid Cube3 action");
        }

        Cube3State next;
        for (int i = 0; i < 8; ++i) {
            const uint8_t source = kActionCornerPerm[action][i];
            next.corner_perm[i] = state_.corner_perm[source];
            next.corner_ori[i] = static_cast<uint8_t>(
                (state_.corner_ori[source] + kActionCornerOri[action][i]) % 3
            );
        }
        for (int i = 0; i < 12; ++i) {
            const uint8_t source = kActionEdgePerm[action][i];
            next.edge_perm[i] = state_.edge_perm[source];
            next.edge_ori[i] = static_cast<uint8_t>(
                (state_.edge_ori[source] + kActionEdgeOri[action][i]) % 2
            );
        }
        return Cube3(next);
    }

    std::array<uint8_t, 54> faceColors() const {
        std::array<uint8_t, 54> colors{};

        for (int face = 0; face < 6; ++face) {
            colors[face * 9 + 4] = static_cast<uint8_t>(face);
        }

        for (std::size_t pos = 0; pos < kCornerFacelets.size(); ++pos) {
            const auto& facelets = kCornerFacelets[pos];
            const auto& piece_facelets = kCornerFacelets[state_.corner_perm[pos]];
            const uint8_t orientation = state_.corner_ori[pos];
            for (int i = 0; i < 3; ++i) {
                colors[facelets[i]] = static_cast<uint8_t>(
                    piece_facelets[(i + orientation) % 3] / 9
                );
            }
        }

        for (std::size_t pos = 0; pos < kEdgeFacelets.size(); ++pos) {
            const auto& facelets = kEdgeFacelets[pos];
            const auto& piece_facelets = kEdgeFacelets[state_.edge_perm[pos]];
            const uint8_t orientation = state_.edge_ori[pos];
            for (int i = 0; i < 2; ++i) {
                colors[facelets[i]] = static_cast<uint8_t>(
                    piece_facelets[(i + orientation) % 2] / 9
                );
            }
        }

        return colors;
    }

    void validate() const {
        validate_permutation(to_vector(state_.corner_perm), 8, "Cube3 corner_perm");
        validate_bounded(to_vector(state_.corner_ori), 8, 3, "Cube3 corner_ori");
        validate_permutation(to_vector(state_.edge_perm), 12, "Cube3 edge_perm");
        validate_bounded(to_vector(state_.edge_ori), 12, 2, "Cube3 edge_ori");
    }

    Cube3State state_;
};

std::string normalize_puzzle_name(const std::string& puzzle_name) {
    std::string normalized;
    normalized.reserve(puzzle_name.size());
    for (unsigned char ch : puzzle_name) {
        if (ch == '_' || ch == '-' || std::isspace(ch)) {
            continue;
        }
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    return normalized;
}

}  // namespace

std::unique_ptr<Environment> createEnvironment(
    const std::string& puzzle_name,
    const std::string& state_key
) {
    const std::string key = normalize_puzzle_name(puzzle_name);
    if (key == "cube3") {
        return std::make_unique<Cube3>(Cube3::fromKey(state_key));
    }

    constexpr const char* prefix = "npuzzle";
    if (key.rfind(prefix, 0) == 0) {
        const std::string dim_text = key.substr(std::string(prefix).size());
        if (dim_text.empty() || !std::all_of(dim_text.begin(), dim_text.end(), [](unsigned char ch) {
                return std::isdigit(ch);
            })) {
            throw std::runtime_error("invalid NPuzzle name: " + puzzle_name);
        }
        const int dim = std::stoi(dim_text);
        if (dim < 2 || dim > 15) {
            throw std::runtime_error("unsupported NPuzzle dimension: " + std::to_string(dim));
        }
        return std::make_unique<NPuzzle>(NPuzzle::fromKey(state_key, static_cast<uint8_t>(dim)));
    }

    throw std::runtime_error("unknown puzzle: " + puzzle_name);
}

}  // namespace deepcube::puzzle
