#!/bin/sh
set -eu

repo_root=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
python="$repo_root/start/.venv/bin/python"
ui="$repo_root/start/catan_lab/vendor/catanatron/ui"

if [ ! -x "$python" ]; then
  echo "Missing the shared Python environment: $python" >&2
  exit 1
fi
if [ ! -x "$ui/node_modules/.bin/vite" ]; then
  echo "Install the pinned UI dependencies first; see catanLabs/Competitive/PLAY.md." >&2
  exit 1
fi

(cd "$ui" && CTRON_API_URL=http://127.0.0.1:5001 npm run start -- --host 127.0.0.1) &
frontend_pid=$!

cleanup() {
  kill "$frontend_pid" 2>/dev/null || true
  wait "$frontend_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$python" "$repo_root/catanLabs/Competitive/play_web.py"
