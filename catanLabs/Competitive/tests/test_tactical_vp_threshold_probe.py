"""Threshold views must change eligibility, not tactical line scoring."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer
from catanatron.models.actions import generate_playable_actions
from catanatron.models.enums import ActionType

from catanLabs.Catanatron.tournament_protocol import install_v5_protocol
from catanLabs.Competitive.tactical_vp_agent import TacticalVPAgent
from catanLabs.Competitive.tactical_vp_threshold_probe import threshold_summary


class TacticalThresholdProbeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        install_v5_protocol()
        cls.config = json.loads((WORKSPACE / "catanLabs/Competitive/configs/tactical_vp_research.json").read_text())

    def _position(self, vp):
        game = Game([RandomPlayer(color) for color in Color], seed=42,
                    vps_to_win=10, discard_limit=7, friendly_robber=False,
                    number_placement="official_spiral")
        for _ in range(16):
            if game.state.current_prompt.name == "PLAY_TURN":
                break
            game.execute(game.playable_actions[0], validate_action=True)
        actor = game.state.current_color()
        prefix = f"P{game.state.color_to_index[actor]}_"
        game.state.player_state[prefix + "VICTORY_POINTS"] = vp
        game.state.player_state[prefix + "ACTUAL_VICTORY_POINTS"] = vp
        game.state.player_state[prefix + "HAS_ROLLED"] = True
        for resource, count in {"WOOD": 0, "BRICK": 4, "SHEEP": 0,
                                "WHEAT": 2, "ORE": 2}.items():
            game.state.player_state[prefix + resource + "_IN_HAND"] = count
        game.playable_actions = generate_playable_actions(game.state)
        return game, actor

    def test_nested_thresholds_share_exact_scan_when_eligible(self):
        game, actor = self._position(5)
        legal = tuple(game.playable_actions)
        end = next(action for action in legal if action.action_type == ActionType.END_TURN)
        before = game.state.random.getstate()
        agents = {gate: TacticalVPAgent(actor, {**self.config, "tactical_min_vp": gate})
                  for gate in (0, 5, 7)}
        choices = {gate: agent.find_override(game, legal, end)
                   for gate, agent in agents.items()}
        self.assertIsNotNone(choices[0])
        self.assertEqual(choices[0], choices[5])
        self.assertIsNone(choices[7])
        self.assertEqual(agents[0].last_scan, agents[5].last_scan)
        self.assertFalse(agents[7].last_scan["eligible"])
        self.assertEqual(game.state.random.getstate(), before)

        game.state.player_state[f"P{game.state.color_to_index[actor]}_VICTORY_POINTS"] = 7
        game.state.player_state[f"P{game.state.color_to_index[actor]}_ACTUAL_VICTORY_POINTS"] = 7
        late = [agent.find_override(game, legal, end) for agent in agents.values()]
        self.assertEqual(late, [late[0]] * 3)

    def test_threshold_summary_tracks_distinct_games_and_vp_buckets(self):
        rows = [
            {"seed": 1, "seat": 0, "turn": 2, "vp": 3, "scan_ms": 2.0,
             "found_line": True, "override": True, "truncated": False,
             "roots": 1, "followups": 2},
            {"seed": 1, "seat": 0, "turn": 2, "vp": 3, "scan_ms": 3.0,
             "found_line": True, "override": True, "truncated": False,
             "roots": 2, "followups": 4},
            {"seed": 1, "seat": 0, "turn": 4, "vp": 5, "scan_ms": 4.0,
             "found_line": True, "override": True, "truncated": True,
             "roots": 3, "followups": 6},
            {"seed": 2, "seat": 1, "turn": 8, "vp": 7, "scan_ms": 5.0,
             "found_line": False, "override": False, "truncated": False,
             "roots": 4, "followups": 8},
        ]
        summaries = {gate: threshold_summary(rows, gate) for gate in (0, 5, 7)}
        self.assertEqual([summaries[gate]["eligible_decisions"] for gate in (0, 5, 7)],
                         [4, 2, 1])
        self.assertEqual([summaries[gate]["games_with_overrides"] for gate in (0, 5, 7)],
                         [1, 1, 0])
        self.assertEqual(summaries[0]["turns_with_overrides"], 2)
        self.assertEqual(summaries[5]["truncated_scans"], 1)


if __name__ == "__main__":
    unittest.main()
