from __future__ import annotations

import unittest

from gamebench.agents.baselines import GreedyAgent, RandomAgent
from gamebench.evaluation.tournament import TournamentConfig, run_tournament
from gamebench.games.connect4 import ConnectFourAdapter


class TournamentTests(unittest.TestCase):
    def test_round_robin_is_deterministic_and_counts_games(self) -> None:
        config = TournamentConfig(games_per_pair=4, seed=12)
        first = run_tournament(ConnectFourAdapter(), [RandomAgent(), GreedyAgent()], config)
        second = run_tournament(ConnectFourAdapter(), [RandomAgent(), GreedyAgent()], config)
        competitive_fields = ("wins", "draws", "losses", "points", "games", "score_rate", "elo")
        for name, row in first["standings"].items():
            self.assertEqual(
                {key: row[key] for key in competitive_fields},
                {key: second["standings"][name][key] for key in competitive_fields},
            )
            self.assertEqual(row["games"], 4)
            self.assertEqual(row["wins"] + row["draws"] + row["losses"], 4)
            self.assertGreater(row["decision_latency_ms"]["moves"], 0)


if __name__ == "__main__":
    unittest.main()
