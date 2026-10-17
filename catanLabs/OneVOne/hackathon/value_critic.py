"""Visible-state features and a small NumPy inference-only value network."""

import numpy as np

from catanatron.models.enums import RESOURCES, DEVELOPMENT_CARDS

VERSION = "early-value-v1"


def encode(game, color):
    state, board = game.state, game.state.board
    enemy = next(c for c in state.colors if c != color)
    own, other = (f"P{state.color_to_index[c]}_" for c in (color, enemy))
    ps = state.player_state
    nodes = sorted(board.map.land_nodes)
    values = [state.num_turns / 100, float(state.current_color() == color)]
    values += [float(state.current_prompt.name == p) for p in ("PLAY_TURN", "BUILD_INITIAL_SETTLEMENT", "BUILD_INITIAL_ROAD", "MOVE_ROBBER", "DISCARD")]
    production = {c: dict.fromkeys(RESOURCES, 0.0) for c in (color, enemy)}
    for node in nodes:
        potential = dict.fromkeys(RESOURCES, 0.0)
        effective = dict.fromkeys(RESOURCES, 0.0)
        for tile in board.map.adjacent_tiles[node]:
            if tile.resource is not None:
                probability = (6 - abs(7 - tile.number)) / 36
                potential[tile.resource] += probability
                if tile != board.map.land_tiles[board.robber_coordinate]:
                    effective[tile.resource] += probability
        building = board.buildings.get(node)
        owner, count = (None, 0) if building is None else (building[0], 2 if building[1] == "CITY" else 1)
        values += [potential[r] * 3 for r in RESOURCES]
        values += [float(owner == color) * count / 2, float(owner == enemy) * count / 2, float(node in board.board_buildable_ids)]
        if owner in production:
            for r in RESOURCES:
                production[owner][r] += effective[r] * count
    for c, prefix in ((color, own), (enemy, other)):
        values += [ps[prefix + r + "_IN_HAND"] / 10 for r in RESOURCES]
        values += [production[c][r] * 2 for r in RESOURCES]
        values += [ps[prefix + "VICTORY_POINTS"] / 10, ps[prefix + "LONGEST_ROAD_LENGTH"] / 15, ps[prefix + "PLAYED_KNIGHT"] / 5]
        values += [len(state.buildings_by_color[c][kind]) / 5 for kind in ("SETTLEMENT", "CITY")]
        values += [len(board.buildable_node_ids(c)) / 10]
    values += [ps[own + d + "_IN_HAND"] / 5 for d in DEVELOPMENT_CARDS]
    values += [sum(ps[other + d + "_IN_HAND"] for d in DEVELOPMENT_CARDS) / 5, ps[own + "ACTUAL_VICTORY_POINTS"] / 10]
    values += [n / 19 for n in state.resource_freqdeck]
    # Roads are public and identify expansion paths that production alone misses.
    edges = sorted({tuple(sorted(edge)) for tile in board.map.land_tiles.values() for edge in tile.edges.values()})
    values += [(1 if board.get_edge_color(edge) == color else -1 if board.get_edge_color(edge) == enemy else 0) for edge in edges]
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite critic observation")
    return result


class Critic:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            if str(data["version"]) != VERSION:
                raise ValueError("Critic feature version mismatch")
            self.mean, self.scale = data["mean"], data["scale"]
            self.w1, self.b1, self.w2, self.b2 = (data[k] for k in ("w1", "b1", "w2", "b2"))
        if self.w1.shape[0] != len(self.mean) or self.w2.shape != (self.w1.shape[1], 1):
            raise ValueError("Invalid critic dimensions")

    def predict(self, features):
        x = (np.asarray(features, dtype=np.float32) - self.mean) / self.scale
        logits = np.maximum(0, x @ self.w1 + self.b1) @ self.w2 + self.b2
        return 1 / (1 + np.exp(-np.clip(logits[:, 0], -20, 20)))
