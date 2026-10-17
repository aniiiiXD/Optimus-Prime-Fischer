#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$here/.cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$here/.python}"
kit="$here/vendor/kit"
mkdir -p "$here/vendor"
if [[ ! -d "$kit/.git" ]]; then
    git clone https://github.com/lyric-hyde/catan-hackathon-agent.git "$kit"
fi
git -C "$kit" checkout --detach 04fffac90e00610f60e0b06d919f262f4b3ed9db
uv venv --allow-existing --python 3.11 "$here/.venv"
uv pip install --python "$here/.venv/bin/python" -r "$here/requirements.txt"
cd "$kit"
PYTHONHASHSEED=0 "$here/.venv/bin/python" -m catan_kit.selfcheck
