"""Small, public-information two-action lookahead over a value-policy prior.

The fair value policy scores every legal root action. Search then expands the
highest-ranked setup moves (roads, bank trades, and resource-granting cards)
to see whether they unlock a useful second action before the turn ends.
No hidden dice, development-card identity, or opponent hand is sampled.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

from catanatron.models.enums import ActionType
from catanLabs.Competitive.fair_value_agent import FairValueAgent
from catanLabs.Transformer.agent import TransformerAgent


SEARCH_ROOTS = {
    ActionType.BUILD_SETTLEMENT,  # Opening settlement is followed by an opening road.
    ActionType.BUILD_ROAD,
    ActionType.MARITIME_TRADE,
    ActionType.PLAY_YEAR_OF_PLENTY,
    ActionType.PLAY_ROAD_BUILDING,
}
FOLLOWUPS = {
    ActionType.BUILD_ROAD,
    ActionType.BUILD_SETTLEMENT,
    ActionType.BUILD_CITY,
    ActionType.BUY_DEVELOPMENT_CARD,
}


class GuidedSearchAgent(FairValueAgent):
    def __init__(self, color, config):
        super().__init__(color, config)
        self.search_roots_per_type = int(config.get("search_roots_per_type", 3))
        self.search_bonus_weight = float(config.get("search_bonus_weight", 1.0))
        self.transformer_roots_per_type = int(config.get("transformer_roots_per_type", 0))
        if self.search_roots_per_type < 0 or not 0 <= self.search_bonus_weight <= 1:
            raise ValueError("Invalid search budget or bonus weight")
        if self.transformer_roots_per_type < 0:
            raise ValueError("Invalid Transformer search budget")
        self.transformer = None
        if self.transformer_roots_per_type:
            model_path = Path(config["transformer_checkpoint"]).resolve(strict=True)
            expected_hash = config["transformer_sha256"]
            if hashlib.sha256(model_path.read_bytes()).hexdigest() != expected_hash:
                raise ValueError("Transformer checkpoint hash mismatch")
            self.transformer = TransformerAgent(color, model_path)

    def _score_action(self, game, action):
        kind = action.action_type
        if kind == ActionType.ROLL:
            return self._roll_value(game)
        if kind == ActionType.BUY_DEVELOPMENT_CARD:
            return self._buy_value(game)
        if kind == ActionType.PLAY_MONOPOLY:
            return self._monopoly_value(game, action.value)
        after = game.copy()
        after.execute(action, validate_action=False)
        return self.value_fn(after, self.color)

    def _continuation_bonus(self, game, action, immediate_score):
        after = game.copy()
        after.execute(action, validate_action=False)
        if after.winning_color() is not None or after.state.current_color() != self.color:
            return 0.0
        second_actions = (a for a in after.playable_actions if a.action_type in FOLLOWUPS)
        best_second = immediate_score
        for second in second_actions:
            # A development purchase stays an unknown card; the fair policy
            # estimates its value without revealing the top of the deck.
            if second.action_type == ActionType.BUY_DEVELOPMENT_CARD:
                score = self._buy_value(after)
            else:
                later = after.copy()
                later.execute(second, validate_action=False)
                score = self.value_fn(later, self.color)
            best_second = max(best_second, score)
        return self.search_bonus_weight * (best_second - immediate_score)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        if playable_actions[0].action_type == ActionType.MOVE_ROBBER:
            return super().decide(game, playable_actions)

        scored = [(self._score_action(game, action), action) for action in playable_actions]
        buckets = {}
        for score, action in scored:
            if action.action_type in SEARCH_ROOTS:
                buckets.setdefault(action.action_type, []).append((score, action))
        bonus = {}
        for options in buckets.values():
            options.sort(key=lambda item: item[0], reverse=True)
            for score, action in options[: self.search_roots_per_type]:
                bonus[action] = self._continuation_bonus(game, action, score)
        if self.transformer is not None:
            prior_logits = self.transformer.score_actions(game, playable_actions)
            by_type = {}
            for index, action in enumerate(playable_actions):
                if action.action_type in SEARCH_ROOTS:
                    by_type.setdefault(action.action_type, []).append(index)
            direct = {action: score for score, action in scored}
            for indices in by_type.values():
                indices.sort(key=lambda index: prior_logits[index], reverse=True)
                for index in indices[: self.transformer_roots_per_type]:
                    action = playable_actions[index]
                    if action not in bonus:
                        bonus[action] = self._continuation_bonus(game, action, direct[action])

        return max(scored, key=lambda item: item[0] + bonus.get(item[1], 0.0))[1]


def create_agent(color, checkpoint, seed):
    del seed
    config = json.loads(Path(checkpoint).read_text()) if checkpoint is not None else {}
    return GuidedSearchAgent(color, config)
