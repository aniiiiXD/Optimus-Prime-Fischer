#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
workspace_root="$(cd "${project_dir}/.." && pwd)"

bash "${workspace_root}/scripts/setup_env.sh"

if command -v npm >/dev/null 2>&1; then
  npm install --prefix "${project_dir}/web"
else
  echo "Node.js/npm not found; skipped the Next.js web app install."
fi

echo "Ready. Activate with: source ${workspace_root}/.venv/bin/activate"
