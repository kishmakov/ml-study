#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cmake -S bool-bench -B build -DBUILD_SHARED_LIBS=ON
cmake --build build
