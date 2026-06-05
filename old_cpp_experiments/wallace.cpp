#include <vector>

#include "wallace.h"

typedef std::vector<int> Column;

void initFirstColumn(Column& column, int bitness) {
    for (int i = 0; i < bitness; i++) {
        column[i] = i + 1;
        column[2 * bitness - 2 - i] = i + 1;
    }
}

bool isTrivial(Column& column) {
    return std::all_of(column.cbegin(), column.cend(),
                       [](int v) { return v <= 2; });
}

int roundsUp(int output) {
    int triples = output / 3;
    int pairs = (output - 3 * triples) / 2;
    return triples + pairs;
}


int remainders(int output) {
    int triples = output / 3;
    int pairs = (output - 3 * triples) / 2;
    int ones = output - 3 * triples - 2 * pairs;
    return triples + pairs + ones;
}

void calculateNext(const Column& from, Column& to, int bitness) {
    for (int i = 0; i + 1 < 2 * bitness; i++) {
        int roundUps = i == 0 ? 0 : roundsUp(from[i - 1]);
        to[i] = roundUps + remainders(from[i]);
    }
}


size_t depthOfWallaceTree(int bitness) {
    const Column zeroColumn(2 * bitness - 1, 0);

    std::vector<Column> columns(0);

    columns.push_back(std::vector(zeroColumn));

    initFirstColumn(columns.back(), bitness);

    while (!isTrivial(columns.back())) {
        columns.push_back(std::vector(zeroColumn));
        size_t size = columns.size();
        auto& from = columns[size - 2];
        auto& to = columns[size - 1];

        calculateNext(from, to, bitness);
    }

    return columns.size();
}
