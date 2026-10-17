"""Adapter for the existing paired OneVOne evaluation harness."""

from catanatron.players.minimax import AlphaBetaPlayer


def create_agent(color, checkpoint=None, seed=None):
    return AlphaBetaPlayer(
        color,
        AlphaBetaPlayer.Params(depth=2, value_fn="base", prunning=False, epsilon=None),
    )
