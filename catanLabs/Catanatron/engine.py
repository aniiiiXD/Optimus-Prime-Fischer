"""Find the one pinned Catanatron source copy used by this workspace."""

from pathlib import Path
import json
import subprocess


WORKSPACE = Path(__file__).resolve().parents[2]
SOURCE_ROOT = WORKSPACE / "start/catan_lab/vendor/catanatron"
PYTHON_ROOT = SOURCE_ROOT / "catanatron"
MANIFEST = WORKSPACE / "start/catan_lab/sources.json"


def source_info() -> dict[str, str]:
    if not (PYTHON_ROOT / "catanatron/game.py").is_file():
        raise FileNotFoundError(
            "Pinned Catanatron checkout is missing. From start/, run "
            "`.venv/bin/python -m catan_lab.setup_sources`."
        )
    expected = json.loads(MANIFEST.read_text())["catanatron"]
    actual = subprocess.check_output(
        ["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected["commit"]:
        raise RuntimeError(f"Catanatron commit changed: expected {expected['commit']}, got {actual}")
    return {"source": expected["url"], "commit": actual, "python_root": str(PYTHON_ROOT)}
