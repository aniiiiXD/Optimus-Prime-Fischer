"""Verify that the shared environment can import the active game projects."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ENV = WORKSPACE_ROOT / ".venv"


def main() -> None:
    if Path(sys.prefix).resolve() != EXPECTED_ENV.resolve():
        raise SystemExit(f"wrong Python environment: {sys.prefix}; expected {EXPECTED_ENV}")
    sys.path.insert(0, str(WORKSPACE_ROOT))
    for name in (
        "numpy",
        "psutil",
        "torch",
        "transformer_lens",
        "connect_four_engine",
        "connect_four_transformer",
        "connect_four_distillation",
        "tic_tac_toe_transformer",
        "tic_tac_toe_shared_transformer",
        "gamebench",
    ):
        import_module(name)
        print(f"OK {name}")
    print(f"Python {sys.version.split()[0]}: {sys.executable}")
    print(f"PyTorch {version('torch')} / NumPy {version('numpy')}")


if __name__ == "__main__":
    main()
