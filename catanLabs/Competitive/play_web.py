"""Run the pinned Catanatron web app with the lab's FairValue incumbent.

From the repository root:

    start/.venv/bin/python catanLabs/Competitive/play_web.py

Then open the pinned Catanatron UI at http://127.0.0.1:3000 and select
"Hyde FairValue (measured incumbent)" as one or more bot seats.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CATANATRON_SOURCE = ROOT / "start/catan_lab/vendor/catanatron/catanatron"
CONFIG_PATH = ROOT / "catanLabs/Competitive/configs/fair_army_finish.json"

# The research runtime keeps both upstream Catanatron variants isolated. This
# web app must load the same pinned v5 package as the FairValue benchmarks.
sys.path[:0] = [str(ROOT), str(CATANATRON_SOURCE)]

from catanatron.models.player import Color, Player  # noqa: E402
from catanatron.params import NoParams  # noqa: E402
from catanatron.registry import REGISTRY  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import install_v5_protocol  # noqa: E402
from catanLabs.Competitive.fair_value_agent import FairValueAgent  # noqa: E402


class HydeFairValue(Player):
    """The lab's measured FairValue incumbent with its promoted config."""

    LABEL = "Hyde FairValue (measured incumbent)"
    Params = NoParams

    def __init__(self, color: Color, params: NoParams | None = None):
        super().__init__(color, params)
        config = json.loads(CONFIG_PATH.read_text())
        self.policy = FairValueAgent(color, config)

    def decide(self, game, playable_actions):
        return self.policy.decide(game, playable_actions)


def create_app():
    """Install the v5 simulator protocol and register FairValue for the UI."""
    install_v5_protocol()
    REGISTRY.register("HYDE_FAIRVALUE", HydeFairValue, replace=True)

    from catanatron.web import create_app as create_catanatron_app

    return create_catanatron_app()


def main() -> None:
    app = create_app()
    print("Catanatron + Hyde FairValue listening on http://127.0.0.1:5001", flush=True)
    app.run(host="127.0.0.1", port=5001, threaded=True)


if __name__ == "__main__":
    main()
