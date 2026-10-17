"""Raw visible records reconstruct the exact canonical tensor rows."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanLabs.Transformer.model import encode_visible_observation  # noqa: E402
from catanLabs.Transformer.raw_storage import read_raw_records  # noqa: E402
from catanLabs.Transformer.visible_observation import (  # noqa: E402
    observation_from_dict, observation_to_dict,
)


class RawStorageTests(unittest.TestCase):
    def test_two_game_collector_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset.npz"
            command = [
                sys.executable, str(WORKSPACE / "catanLabs/Transformer/collect.py"),
                "--games", "2", "--max-examples", "1000", "--wall-seconds", "25",
                "--seed", "20266301", "--output", str(output),
            ]
            subprocess.run(command, cwd=WORKSPACE, check=True, capture_output=True,
                           text=True, timeout=35)
            raw_path = output.with_name("dataset.raw.jsonl.gz")
            manifest = json.loads(output.with_suffix(".json").read_text())
            rows = list(read_raw_records(raw_path))
            tensor = np.load(output, allow_pickle=False)
            self.assertEqual(manifest["raw_records"], len(rows))
            self.assertEqual(len(rows), len(tensor["labels"]))
            self.assertGreater(len(rows), 100)
            self.assertEqual(len(manifest["games"]), 2)
            self.assertTrue(all(game["status"] == "completed" for game in manifest["games"]))
            seat_colors = {}
            for game_info in manifest["games"]:
                game = Game([RandomPlayer(color) for color in Color], seed=game_info["seed"],
                            vps_to_win=10, discard_limit=7, friendly_robber=False,
                            number_placement="official_spiral")
                seat_colors[game_info["seed"]] = tuple(color.value for color in game.state.colors)
            self.assertEqual(hashlib.sha256(raw_path.read_bytes()).hexdigest(), manifest["raw_sha256"])
            for index, raw in enumerate(rows):
                self.assertEqual(observation_from_dict(
                    json.loads(json.dumps(observation_to_dict(raw.observation))),
                ), raw.observation)
                self.assertEqual(raw.target_action_index, int(tensor["labels"][index]))
                self.assertEqual(raw.game_id, int(tensor["game_ids"][index]))
                self.assertIn(raw.actor_seat, range(4))
                self.assertEqual(raw.actor_color, str(tensor["actor_colors"][index]))
                self.assertEqual(raw.actor_color, seat_colors[raw.board_seed][raw.actor_seat])
                self.assertEqual(raw.phase, str(tensor["phases"][index]))
                self.assertEqual(raw.outcome, int(tensor["outcomes"][index]))
                self.assertEqual(raw.final_vp, int(tensor["final_vps"][index]))
                self.assertEqual(raw.benchmark_protocol, manifest["benchmark_protocol"])
                self.assertEqual(raw.rules_manifest_sha256, manifest["rules_manifest_sha256"])
                self.assertEqual(raw.feature_schema, manifest["feature_schema"])
                arrays = encode_visible_observation(raw.observation)
                for key, value in zip(("state_features", "state_kinds", "state_ids",
                                       "action_features", "action_types", "target_groups",
                                       "target_ids"), arrays):
                    expected = tensor[key][index]
                    if key.startswith("action_") or key.startswith("target_"):
                        expected = expected[:len(raw.observation.legal_actions)]
                    np.testing.assert_array_equal(value, expected, err_msg=f"row {index}: {key}")


if __name__ == "__main__":
    unittest.main()
