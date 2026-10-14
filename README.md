# Hyde game-learning workspace

CPU-first experiments in game engines, small Transformers, training, evaluation,
and profiling. Start with the [project map](start/README.md); the shared Python
setup is in `start/scripts/setup_env.sh`.

The repository keeps source code, study notes, reports, and the small trained
checkpoints needed to play. It deliberately omits the Python environment,
Node build/install directories, large training datasets and caches, downloaded
papers/reference archives, and two independent upstream Git checkouts. Those
resources remain on the current machine; they are **not** deleted. See
[external resources and restoration](docs/EXTERNAL_RESOURCES.md) before using
the Othello lab, the Connect Four exact solver, or the knowledge corpus from a
fresh clone.

```bash
cd start
bash scripts/setup_env.sh
bash scripts/check_env.sh
.venv/bin/python -m connect_four_transformer.play --human X \
  --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v5_5min.pt
```

This is an experimental workspace, not a claim that any model plays perfectly.
Training is opt-in; setup and the command above do not start training.
