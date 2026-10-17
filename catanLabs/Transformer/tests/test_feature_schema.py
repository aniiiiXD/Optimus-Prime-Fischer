"""Regression checks for the versioned visible-state token layout."""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE))
sys.path.insert(0, str(WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"))

from catanatron import Color, Game, RandomPlayer

from catanLabs.Transformer.agent import TransformerAgent
from catanLabs.Transformer.model import (
    ACTION_FEATURE_DIM,
    DATASET_SCHEMA,
    FEATURE_DIM,
    FEATURE_SCHEMA,
    RESOURCES,
    STATE_TOKEN_COUNT,
    encode_observation,
)
from catanLabs.Transformer.train import validate_dataset_schema


class FeatureSchemaTests(unittest.TestCase):
    def test_all_port_types_have_distinct_node_bits(self):
        game = Game(
            [RandomPlayer(color) for color in Color], seed=42, vps_to_win=10,
            discard_limit=7, friendly_robber=False,
            number_placement="official_spiral",
        )
        state_features = encode_observation(game, list(game.playable_actions))[0]
        first_node_token = 1 + 4 + 19
        port_nodes = game.state.board.map.port_nodes

        for port_type, nodes in port_nodes.items():
            category = 1 if port_type is None else 2 + RESOURCES.index(port_type)
            for node in nodes:
                encoded = state_features[first_node_token + node, 12:19]
                self.assertEqual(float(encoded.sum()), 1.0)
                self.assertEqual(int(encoded.argmax()), category)

        all_port_nodes = set().union(*port_nodes.values())
        no_port = next(node for node in range(54) if node not in all_port_nodes)
        self.assertEqual(int(state_features[first_node_token + no_port, 12:19].argmax()), 0)
        self.assertEqual(float(state_features[first_node_token + no_port, 12:19].sum()), 1.0)

        actor = game.state.current_color()
        buildable = set(game.state.board.buildable_node_ids(actor))
        for node in range(54):
            row = state_features[first_node_token + node]
            self.assertEqual(row[19], float(node in buildable))
            self.assertEqual(row[20], len(game.state.board.map.adjacent_tiles.get(node, ())) / 3.0)

    def test_old_or_missing_dataset_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npz"
            arrays = {
                "state_features": np.zeros((1, STATE_TOKEN_COUNT, FEATURE_DIM), dtype=np.float32),
                "action_features": np.zeros((1, 1, ACTION_FEATURE_DIM), dtype=np.float32),
            }
            np.savez(path, **arrays)
            with np.load(path, allow_pickle=False) as raw:
                with self.assertRaisesRegex(ValueError, "Regenerate"):
                    validate_dataset_schema(raw)

            np.savez(path, dataset_schema="catanlabs-transformer-dataset-v1",
                     feature_schema="visible-state-tokens-v1", **arrays)
            with np.load(path, allow_pickle=False) as raw:
                with self.assertRaisesRegex(ValueError, "Regenerate"):
                    validate_dataset_schema(raw)

            np.savez(path, dataset_schema=DATASET_SCHEMA,
                     feature_schema=FEATURE_SCHEMA, **arrays)
            with np.load(path, allow_pickle=False) as raw:
                validate_dataset_schema(raw)

    def test_old_checkpoint_is_rejected_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "old.pt"
            for schema, feature_schema in (
                ("catanlabs-action-transformer-v1", "visible-state-tokens-v1"),
                ("catanlabs-action-transformer-v2", "visible-state-tokens-v1"),
            ):
                torch.save({"schema": schema, "feature_schema": feature_schema}, path)
                with self.assertRaisesRegex(ValueError, "Incompatible"):
                    TransformerAgent(None, path)


if __name__ == "__main__":
    unittest.main()
