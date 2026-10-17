"""Bridge the paper author's adapter API to this workspace's pinned engine.

The author's adapter imports catanatron_experimental, which is a separate
package/version. These names are direct equivalents in pinned Catanatron.
"""

from catanatron import Color, Game
from catanatron.models.player import Player
from catanatron.players.tree_search_utils import (
    execute_deterministic,
    execute_spectrum,
    expand_spectrum,
    list_prunned_actions,
    prune_robber_actions,
)
from catanatron.players.value import base_fn, get_value_fn, value_production


def copy_game(game):
    return game.copy()
