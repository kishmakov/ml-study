#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT_DIR}/daemon/worker"
CXX="clang++"

"${CXX}" \
    -std=c++17 \
    -O2 \
    -pthread \
    -I"${ROOT_DIR}" \
    "${ROOT_DIR}/daemon/worker.cpp" \
    "${ROOT_DIR}/puzzle/environment.cpp" \
    -lzmq \
    -o "${OUT}"

chmod +x "${OUT}"
printf 'Built native worker: %s\n' "${OUT}"
