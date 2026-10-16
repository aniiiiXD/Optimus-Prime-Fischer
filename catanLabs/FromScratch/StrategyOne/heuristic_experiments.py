"""Explicit, optional gameplay heuristics for isolated ablation tests."""

from __future__ import annotations

from typing import Any


DICE_PIPS = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}


def settlement_pips(observation: dict[str, Any], node: int) -> int:
    """Expected production count of a candidate node, ignoring future trades."""
    return sum(
        DICE_PIPS.get(tile["dice_number"], 0)
        for tile in observation["board"]["tiles"]
        if node in tile["nodes"] and tile["resource"] is not None
    )


def own_hand_size(observation: dict[str, Any]) -> int:
    return sum(int(count) for count in observation["own_hand"].values())
