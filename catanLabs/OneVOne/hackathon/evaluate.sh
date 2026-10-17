#!/usr/bin/env bash
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
nano="$(cd -- "$here/../../.." && pwd)/my_team/nano.py"
cd "$here/vendor/kit"
export PYTHONHASHSEED=0
exec "$here/.venv/bin/python" -m catan_kit.evaluate \
    --a "$nano" --b examples/f_agent.py --games 20 "$@"
