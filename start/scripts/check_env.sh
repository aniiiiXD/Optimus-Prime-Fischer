#!/usr/bin/env bash
set -euo pipefail
workspace_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$workspace_root/.venv/bin/python" "$workspace_root/scripts/check_env.py"
