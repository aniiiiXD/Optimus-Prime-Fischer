# Game-learning workspace

This workspace has one shared Python environment at **`start/.venv/`**. Each
game and experiment keeps its own code, checkpoints, reports, and datasets.
The original Othello research checkout is a separate repository; it is not the
parent of the Tic-Tac-Toe or Connect Four projects.

| Folder | Purpose |
| --- | --- |
| [`tic_tac_toe_lab/`](tic_tac_toe_lab/README.md) | Rules, Q-learning, and a CPU MLP |
| [`tic_tac_toe_transformer/`](tic_tac_toe_transformer/README.md) | Token/attention Transformer and training |
| [`tic_tac_toe_shared_transformer/`](tic_tac_toe_shared_transformer/README.md) | Shared decision-style Transformer comparison |
| [`connect_four_engine/`](connect_four_engine/README.md) | Independent game rules and terminal play |
| [`connect_four_transformer/`](connect_four_transformer/README.md) | Models, datasets, checkpoints, evaluation, browser UI |
| [`connect_four_distillation/`](connect_four_distillation/README.md) | Isolated strong-solver teacher experiment |
| [`othello_pipeline/`](othello_pipeline/README.md) | Neural-engine teacher, CPU Othello Transformer, pilot and evaluation |
| [`catan_lab/`](catan_lab/README.md) | Pinned Catan agents, pretrained CPU pilots, rule/architecture audit and benchmark harness |
| `othello_world/` | Optional upstream OthelloGPT checkout and local study lab; [not included in this repository](../docs/EXTERNAL_RESOURCES.md#restore-othello-research-checkout) |
| [`transformer_games_lab/`](transformer_games_lab/README.md) | Reusable game adapters, search, tournaments, CPU profiler, Next.js UI |
| [`game_transformer_knowledge_corpus/`](game_transformer_knowledge_corpus/README.md) | Research notes and source catalog; large downloads are optional |
| [`model_styles_catalog/`](model_styles_catalog/README.md) | Notes comparing model/decision styles |

## One Python environment

From `start/`, run `bash scripts/setup_env.sh` to create/update `.venv` using
the direct dependency pins in [`requirements.txt`](requirements.txt). The
setup uses `uv` and Python 3.11; `bash scripts/check_env.sh` verifies key
imports and reports the interpreter path. To enter it interactively:

```bash
source .venv/bin/activate
```

Or skip activation and run `.venv/bin/python` from the workspace root. There
is no `.venv` inside any game directory; do not create a fresh one there.

The Next.js app under `transformer_games_lab/web/` has its **own Node.js
dependencies** in `node_modules/`; Python venvs cannot contain npm packages.
On a CUDA machine, Othello's optional CUDA setup replaces the PyTorch build
in this **same** `.venv`; it does not create another environment.

## Quick commands from `start/`

```bash
.venv/bin/python -m connect_four_transformer.play --human X --checkpoint connect_four_transformer/artifacts/checkpoints/candidate_tactical_v5_5min.pt
.venv/bin/python -m unittest connect_four_distillation.test_oracle
.venv/bin/python -m tic_tac_toe_transformer.play
```

The exact-solver oracle command requires the [separate Pascal Pons checkout](../docs/EXTERNAL_RESOURCES.md#restore-the-solver).
Large Connect Four training data and teacher caches are not in Git, but the
small playable checkpoints and evaluation reports are.

In the **original local workspace**, Othello's study lab can use the shared
interpreter from its own working directory:

```bash
cd othello_world
../.venv/bin/python lab/app.py
```

The Othello checkout and its local lab additions are not in the pushed
repository; [read the restoration caveat](../docs/EXTERNAL_RESOURCES.md#restore-othello-research-checkout).

See each project's README for training, profiling, browser UI, and checkpoint
details. All long-running training remains opt-in; setting up the environment
does not start a training run.
