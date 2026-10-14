#!/usr/bin/env bash
set -euo pipefail

workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace_root"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required; install it before running this setup." >&2
  exit 1
fi

uv venv --allow-existing --python 3.11 "$workspace_root/.venv"
uv pip install --python "$workspace_root/.venv/bin/python" -r "$workspace_root/requirements.txt"
"$workspace_root/.venv/bin/python" "$workspace_root/scripts/check_env.py"
