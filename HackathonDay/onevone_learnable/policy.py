"""Legal candidates -> expert prior + replaceable learned value -> win guard.

Rules determine legality. Strategic weights are soft. Any custom learner may
score the same full legal set, or replace selection without changing the engine.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import process_time
import numpy as np
from expert import HeuristicExpert
from features import SCHEMA, encode_board
from runtime import NeuralScorer
from tactics import winning_root

ROOT = Path(__file__).resolve().parent


def load_config(path=None):
    return json.loads((Path(path) if path else ROOT/"config.json").read_text())


class LearnableOneVOne:
    def __init__(self, color, config=None, scorer=None):
        self.color = color
        self.config = load_config() if config is None else dict(config)
        if self.config.get("schema") != SCHEMA:
            raise ValueError("Policy feature schema mismatch")
        self.learning_weight = float(self.config.get("learning_weight", .35))
        self.prior_weight = float(self.config.get("prior_weight", 1.0))
        if not np.isfinite(self.learning_weight) or self.learning_weight < 0:
            raise ValueError("learning_weight must be finite and nonnegative")
        if not np.isfinite(self.prior_weight) or self.prior_weight < 0:
            raise ValueError("prior_weight must be finite and nonnegative")
        self.expert = HeuristicExpert(color, self.config)
        self.scorer = scorer
        arch = self.config.get("scorer", "heuristic")
        if arch not in ("heuristic", "mlp", "transformer"):
            raise ValueError(f"Unknown scorer: {arch}")
        if self.scorer is None and arch != "heuristic":
            checkpoint = self.config.get("checkpoint")
            if not checkpoint:
                raise ValueError("A neural scorer requires a checkpoint")
            path = Path(checkpoint)
            self.scorer = NeuralScorer(path if path.is_absolute() else ROOT/path, arch)
        self.last_decision = None
        self.stats = {"decisions": 0, "model_calls": 0, "model_changes": 0,
                      "win_guards": 0, "clock_fallbacks": 0, "model_errors": 0}

    def prepare_candidates(self, game, legal_actions):
        return self.expert.prepare(game, legal_actions)

    def select_candidate(self, board, candidates):
        """Override this one hook for an MLP, transformer, RL policy or ensemble.

        Equal-shape finite scores are mandatory. A failed model uses the expert
        for this decision and records the failure instead of choosing randomly.
        """
        prior = np.asarray([c.rank for c in candidates], dtype=np.float32)
        logits = None
        error = None
        if self.scorer is not None and self.learning_weight > 0:
            try:
                logits = np.asarray(self.scorer.predict(board, np.stack([c.features for c in candidates])), dtype=np.float32)
                if logits.shape != prior.shape or not np.isfinite(logits).all():
                    raise ValueError("Model must return one finite logit per legal action")
                # Bounded learned residual has a configurable influence. A
                # weight above 1 can overturn the full expert ordering.
                combined = self.prior_weight*prior + self.learning_weight*np.tanh(logits-np.median(logits))
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                error = repr(exc)
                combined = prior
        else:
            combined = prior
        return int(np.argmax(combined)), logits, combined, error

    @staticmethod
    def fast_choice(actions):
        # Reserve-clock behavior is explicit and deterministic. The engine's
        # supplied list still defines every permitted action.
        order = {"BUILD_CITY": 0, "BUILD_SETTLEMENT": 1, "PLAY_KNIGHT_CARD": 2,
                 "ROLL": 3, "END_TURN": 4}
        return min(range(len(actions)), key=lambda i: order.get(actions[i].action_type.name, 10))

    def decide(self, game, actions, cpu_left=None):
        if not actions:
            raise ValueError("No legal actions supplied")
        start = process_time()
        self.stats["decisions"] += 1
        if len(actions) == 1:
            self.last_decision = {"chosen": 0, "reason": "forced", "candidates": []}
            return 0
        if cpu_left is not None and cpu_left < .06:
            choice = self.fast_choice(actions)
            self.stats["clock_fallbacks"] += 1
            self.last_decision = {"chosen": choice, "reason": "clock_reserve", "candidates": []}
            return choice
        _, candidates = self.prepare_candidates(game, actions)
        prior_choice = max(range(len(candidates)), key=lambda i: candidates[i].rank)
        board = encode_board(game, self.color) if self.scorer is not None else None
        choice, logits, combined, error = self.select_candidate(board, candidates)
        if logits is not None and error is None:
            self.stats["model_calls"] += 1
        self.stats["model_errors"] += error is not None
        total_budget = min(self.config.get("max_decision_ms", 9)/1000,
                           max(0, (cpu_left-.05)*.2) if cpu_left is not None else .009)
        deadline = min(start+total_budget, process_time()+self.config.get("tactical_ms", 2)/1000)
        guard, nodes = winning_root(game, candidates, self.color, deadline,
                                    max_nodes=self.config.get("tactical_nodes", 96))
        if guard is not None:
            choice = guard
            self.stats["win_guards"] += 1
        if logits is not None and error is None:
            self.stats["model_changes"] += choice != (guard if guard is not None else prior_choice)
        self.last_decision = {"chosen": choice, "prior_choice": prior_choice,
                              "reason": "guaranteed_win" if guard is not None else "scored",
                              "model_error": error, "logits": logits, "combined": combined,
                              "search_nodes": nodes, "candidates": candidates, "board": board}
        return choice
