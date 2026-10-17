#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$here/vendor/kit"
export PYTHONHASHSEED=0
exec "$here/.venv/bin/python" -m catan_kit.evaluate \
    --a "$here/agent.py" --b examples/f_agent.py --games 20 "$@"
