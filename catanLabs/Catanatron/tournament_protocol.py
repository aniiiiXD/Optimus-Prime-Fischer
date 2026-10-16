"""Shared Catanatron v5 chance and action-order contract.

Install before creating a game or asking an alpha-beta teacher to score moves.
The pinned upstream Game.copy shares the live RNG, and Game.execute ignores
explicit hypothetical roll/card outcomes unless they arrive as ActionRecord.
"""

from __future__ import annotations

import random


PROTOCOL = "catanLabs-four-player-v5-canonical-actions-and-chance"


def canonical_action_key(action):
    """Stable tie-breaking independent of Python object hashes."""
    return (action.action_type.name, repr(action.value), action.color.value)


def install_v5_protocol() -> None:
    """Patch the pinned upstream Game once, preserving ordinary chance draws."""
    from catanatron import Game
    from catanatron.models.enums import ActionRecord, ActionType

    if getattr(Game, "_catanlabs_v5_protocol", False):
        return

    original_copy = Game.copy
    original_execute = Game.execute

    def copy_with_isolated_rng(game):
        clone = original_copy(game)
        copied_rng = random.Random()
        copied_rng.setstate(game.state.random.getstate())
        clone.state.random = copied_rng
        clone.random = copied_rng
        return clone

    def execute_with_forced_search_outcome(game, action, validate_action=True, action_record=None):
        if (
            not validate_action
            and action_record is None
            and action.value is not None
            and action.action_type in (ActionType.ROLL, ActionType.BUY_DEVELOPMENT_CARD)
        ):
            action_record = ActionRecord(action, action.value)
        result = original_execute(game, action, validate_action, action_record)
        game.playable_actions.sort(key=canonical_action_key)
        return result

    Game.copy = copy_with_isolated_rng
    Game.execute = execute_with_forced_search_outcome
    Game._catanlabs_v5_protocol = True
