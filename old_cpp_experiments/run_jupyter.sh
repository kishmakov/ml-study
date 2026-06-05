#!/usr/bin/env bash

set -euo pipefail

# Use the directory of this script as Jupyter root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Allow overriding port via env var; default to 8888
PORT="${PORT:-8888}"

VENV_ACTIVATE="$HOME/jupyter-env/bin/activate"
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Virtual environment activate script not found: $VENV_ACTIVATE" >&2
  exit 1
fi

source "$VENV_ACTIVATE"

cd "$SCRIPT_DIR"

if command -v jupyter-lab >/dev/null 2>&1; then
  # JupyterLab v3+: configure root_dir via ServerApp
  exec jupyter lab \
    --ip=127.0.0.1 \
    --port "$PORT" \
    --ServerApp.open_browser=false \
    --ServerApp.root_dir="$SCRIPT_DIR" \
    --IdentityProvider.token='' \
    --ServerApp.disable_check_xsrf=True \
    --ServerApp.password=''
else
  # Classic Notebook
  exec jupyter notebook \
    --port "$PORT" \
    --notebook-dir="$SCRIPT_DIR" \
    --NotebookApp.open_browser=true
fi
