#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_PROJECT="${HOME}/Repos/shad/shad_2026_2"

# Reuses the uv environment (torch, numpy, ...) of the shad_2026_2 project.
exec uv run --project "${UV_PROJECT}" python "${ROOT_DIR}/test.py" "$@"
