#include "solution.h"

#include <cassert>
#include <vector>


struct SelectedBits
{
    uint64_t mask;
    uint64_t values;

    [[nodiscard]] uint64_t MaskComplement(uint64_t N) const {
        return ((1ull << N) - 1) ^ mask;
    }
};

void BuildNodes(
    uint64_t N,
    const std::string& values,
    size_t currentId,
    SelectedBits bits,
    std::vector<Node>& nodes
) {
    int seen = 0;

    uint64_t mask_complement = bits.MaskComplement(N);
    uint64_t sub = mask_complement;
    do {
        uint64_t id = sub | bits.values;
        seen |= 1 << (values >> id);
        sub = (sub - 1) & mask_complement;
    } while (sub != mask_complement);

    assert(seen > 0);

    if (seen == 3) {
        uint64_t l0 = nodes.size();
        uint64_t l1 = nodes.size() + 1;

        uint64_t bitId = 0;
        while ((1ull << bitId) & bits.mask) {
            bitId++;
        }

        nodes[currentId].division = {bitId, l0, l1};
        nodes.emplace_back();
        nodes.emplace_back();

        SelectedBits bits0 = bits;
        SelectedBits bits1 = bits;

        bits0.mask |= 1ull << bitId;
        bits1.mask |= 1ull << bitId;
        bits1.values |= 1ull << bitId;

        BuildNodes(N, values, l0, bits0, nodes);
        BuildNodes(N, values, l1, bits1, nodes);
    } else {
        nodes[currentId].value = seen == 2;
    }
}

std::vector<Node> Solve(uint64_t N, const std::string& values) {
    std::vector<Node> nodes(0);
    nodes.emplace_back(std::nullopt, false);
    BuildNodes(N, values, 0, {0, 0}, nodes);
    return nodes;
}
