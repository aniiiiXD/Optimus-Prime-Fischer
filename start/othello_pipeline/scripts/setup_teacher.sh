#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEACHER_REPO="$PROJECT_ROOT/vendor/neural-reversi"
WEIGHT_DIR="$PROJECT_ROOT/artifacts/teacher_weights"
PINNED_COMMIT=2d5e430647c16bea2033c2aeaeda138c96ed6b07

if [[ ! -d "$TEACHER_REPO/.git" ]]; then
  git clone --depth 1 --branch v6.3.0 https://github.com/natsutteatsuiyone/neural-reversi.git "$TEACHER_REPO"
fi
if [[ "$(git -C "$TEACHER_REPO" rev-parse HEAD)" != "$PINNED_COMMIT" ]]; then
  echo "Teacher checkout is not the pinned v6.3.0 commit: $TEACHER_REPO" >&2
  exit 1
fi

mkdir -p "$WEIGHT_DIR"
download_weight() {
  local file_name="$1"
  local expected_sha="$2"
  local destination="$WEIGHT_DIR/$file_name"
  if [[ ! -f "$destination" ]]; then
    curl -fL --retry 3 --connect-timeout 15 \
      --output "$destination.part" \
      "https://github.com/natsutteatsuiyone/neural-reversi-weights/releases/download/v7.0.0-wasm/$file_name"
    mv "$destination.part" "$destination"
  fi
  local actual_sha
  actual_sha="$(shasum -a 256 "$destination" | awk '{print $1}')"
  if [[ "$actual_sha" != "$expected_sha" ]]; then
    echo "SHA-256 mismatch for $destination; expected $expected_sha, got $actual_sha" >&2
    exit 1
  fi
  local link="$TEACHER_REPO/$file_name"
  if [[ -e "$link" || -L "$link" ]]; then
    if [[ "$(realpath "$link")" != "$(realpath "$destination")" ]]; then
      echo "Unexpected existing teacher weight link/file: $link" >&2
      exit 1
    fi
  else
    ln -s "../../artifacts/teacher_weights/$file_name" "$link"
  fi
  echo "verified $file_name"
}

download_weight eval-e6bbc4f6.zst bfc7e581500bc20d6fbe77d0155f988e6c010d407470bb1cfca4ceb3ffe95c41
download_weight eval_sm-e6bbc4f6.zst af43393d3c0b92132a3f422e7c40f75089fa8aa96725c108dc3e50fcc9dd5041

(cd "$TEACHER_REPO" && cargo build --release -p cli)
echo "teacher CLI ready: $TEACHER_REPO/target/release/cli"
