#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${ROOT_DIR}/daemon/worker"
CXX="clang++"
TORCH_DIR="${ROOT_DIR}/.venv/lib/python3.12/site-packages/torch"
TORCH_LIB="${TORCH_DIR}/lib"

"${CXX}" \
    -std=c++17 \
    -O2 \
    -pthread \
    -D_GLIBCXX_USE_CXX11_ABI=1 \
    -I"${ROOT_DIR}" \
    -I"${TORCH_DIR}/include" \
    -I"${TORCH_DIR}/include/torch/csrc/api/include" \
    "${ROOT_DIR}/daemon/worker.cpp" \
    "${ROOT_DIR}/costtogo.cpp" \
    "${ROOT_DIR}/puzzle/environment.cpp" \
    "${ROOT_DIR}/search/a_star.cpp" \
    -L"${TORCH_LIB}" \
    -Wl,-rpath,"${TORCH_LIB}" \
    -ltorch \
    -ltorch_cpu \
    -lc10 \
    -lzmq \
    -o "${OUT}"

chmod +x "${OUT}"
printf 'Built native worker: %s\n' "${OUT}"
