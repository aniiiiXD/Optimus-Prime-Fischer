"""Expose the current OneVOne policy in the pinned Catanatron human-play UI.

From the workspace root, run ``start/.venv/bin/python
catanLabs/OneVOne/play_web.py`` and open http://127.0.0.1:3000.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATANATRON_SOURCE = ROOT / "start/catan_lab/vendor/catanatron/catanatron"
sys.path[:0] = [str(ROOT), str(CATANATRON_SOURCE)]

from catanatron.models.player import Color  # noqa: E402
from catanatron.params import NoParams  # noqa: E402
from catanatron.registry import REGISTRY  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import install_v5_protocol  # noqa: E402
from catanLabs.OneVOne.agent import DEFAULT_CONFIG, TRAINED_CONFIG, OneVOneAgent  # noqa: E402


class HydeOneVOne(OneVOneAgent):
    """The trained 1v1 policy, falling back to the measured default."""

    LABEL = "Hyde OneVOne"
    Params = NoParams

    def __init__(self, color: Color, params: NoParams | None = None):
        del params
        config_path = TRAINED_CONFIG if TRAINED_CONFIG.is_file() else DEFAULT_CONFIG
        config = json.loads(config_path.read_text())
        super().__init__(color, config)


def create_app():
    """Install the local simulator protocol and register this player."""
    install_v5_protocol()
    REGISTRY.register("HYDE_ONEVONE", HydeOneVOne, replace=True)

    from catanatron.web import create_app as create_catanatron_app

    return create_catanatron_app()


def main() -> None:
    app = create_app()
    print("Catanatron + Hyde OneVOne listening on http://127.0.0.1:5001", flush=True)
    app.run(host="127.0.0.1", port=5001, threaded=True)


if __name__ == "__main__":
    main()
