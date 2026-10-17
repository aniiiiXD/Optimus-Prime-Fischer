"""Pluggable NumPy inference; game workers do not need PyTorch.

A custom scorer implements predict(board[55,32], candidates[N,D]) -> logits[N].
Both supplied networks predict eventual wins from visible state/action features.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from features import CANDIDATE_DIM, SCHEMA


class NeuralScorer:
    def __init__(self, checkpoint, expected_arch=None):
        with np.load(Path(checkpoint), allow_pickle=False) as data:
            self.schema = str(data["schema"].item())
            self.arch = str(data["architecture"].item())
            self.weights = {k: data[k].astype(np.float32) for k in data.files
                            if k not in ("schema", "architecture")}
        if self.schema != SCHEMA or self.arch not in ("mlp", "transformer"):
            raise ValueError(f"Unsupported checkpoint: {self.schema}/{self.arch}")
        if expected_arch is not None and self.arch != expected_arch:
            raise ValueError(f"Expected {expected_arch}, got {self.arch}")
        expected = CANDIDATE_DIM + 64 if self.arch == "mlp" else CANDIDATE_DIM
        first = "mlp1.weight" if self.arch == "mlp" else "candidate.weight"
        if self.weights[first].shape[1] != expected:
            raise ValueError("Checkpoint candidate feature dimension mismatch")

    def linear(self, x, name):
        return x @ self.weights[name+".weight"].T + self.weights[name+".bias"]

    def norm(self, x, name):
        centered = x - x.mean(axis=-1, keepdims=True)
        normalized = centered / np.sqrt((centered*centered).mean(axis=-1, keepdims=True)+1e-5)
        return normalized*self.weights[name+".weight"] + self.weights[name+".bias"]

    def encode(self, board):
        x = self.linear(board, "project") + self.weights["position"]
        count, width = x.shape
        heads = len(self.weights["edge_bias"])
        q, k, v = self.linear(self.norm(x, "norm1"), "qkv").reshape(count, 3, heads, width//heads).transpose(1, 2, 0, 3)
        scores = q @ k.transpose(0, 2, 1) / np.sqrt(width//heads)
        scores += self.weights["edge_bias"][:, None, None]*self.weights["adjacency"][None]
        exp = np.exp(scores-scores.max(axis=-1, keepdims=True))
        joined = ((exp/exp.sum(axis=-1, keepdims=True)) @ v).transpose(1, 0, 2).reshape(count, width)
        x = x + self.linear(joined, "attn_out")
        x = x + self.linear(np.maximum(self.linear(self.norm(x, "norm2"), "ff1"), 0), "ff2")
        return self.norm(x, "norm3")

    @np.errstate(over="ignore", invalid="ignore", divide="ignore")
    def predict(self, board, candidates):
        board, candidates = np.asarray(board, dtype=np.float32), np.asarray(candidates, dtype=np.float32)
        if board.shape != (55, 32) or candidates.ndim != 2 or candidates.shape[1] != CANDIDATE_DIM:
            raise ValueError("Scorer input shape mismatch")
        if self.arch == "mlp":
            context = np.concatenate((board[:54].mean(axis=0), board[-1]))
            x = np.concatenate((candidates, np.broadcast_to(context, (len(candidates), 64))), axis=-1)
            out = self.linear(np.maximum(self.linear(np.maximum(self.linear(x, "mlp1"), 0), "mlp2"), 0), "head")
        else:
            tokens = self.encode(board)
            marked = candidates[:, -54:] @ tokens[:54]
            context = np.broadcast_to(tokens[-1], (len(candidates), len(tokens[-1])))
            encoded = np.maximum(self.linear(candidates, "candidate"), 0)
            x = np.concatenate((context, marked, encoded), axis=-1)
            out = self.linear(np.maximum(self.linear(x, "head1"), 0), "head2")
        out = out.reshape(-1)
        if not np.isfinite(out).all():
            raise FloatingPointError("Nonfinite model logits")
        return out
