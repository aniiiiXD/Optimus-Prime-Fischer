# Local-only resources and how to restore them

The root `.gitignore` excludes external checkouts and bulky/generated files.
They remain available in the original workspace but will **not** appear in a
GitHub clone. Clone external repositories into the paths below; do not `git add`
them to the Hyde repository, because that would record a gitlink rather than
their contents. The commits below describe the current local checkouts.

| Local path under `start/` | Source or regeneration | Why it is omitted |
| --- | --- | --- |
| `othello_world/` | [likenneth/othello_world](https://github.com/likenneth/othello_world), commit `f23bb5696cf30b93bd8af8a391ee33fc3aac417e` | Independent Git repository, plus a local pretrained checkpoint |
| `connect_four_distillation/vendor/connect4/` | [PascalPons/connect4](https://github.com/PascalPons/connect4), commit `d6ba50d8aaf2308c769d9bf2abd42d90f34baf41` | Independent AGPL-3.0 solver checkout and locally built binary |
| `game_transformer_knowledge_corpus/repos/`, `papers/`, `archives/` | [source catalog](../start/game_transformer_knowledge_corpus/sources.json), [download manifest](../start/game_transformer_knowledge_corpus/download_manifest.json) | Downloaded research materials and large files |
| `connect_four_transformer/artifacts/data/`, `cache/` | Dataset generators and teacher-cache code in `connect_four_transformer/` | Regenerable training examples and SQLite cache |
| `.venv/`, `transformer_games_lab/web/node_modules/`, `web/.next/` | `scripts/setup_env.sh` and `npm ci` in the web folder | Reinstallable dependencies/build output |

Small model checkpoints and evaluation reports under `start/*/artifacts/` are
kept in Git so the provided play and benchmark commands have weights to load.

## Restore the solver

From `start/`:

```bash
git clone https://github.com/PascalPons/connect4.git connect_four_distillation/vendor/connect4
git -C connect_four_distillation/vendor/connect4 checkout d6ba50d8aaf2308c769d9bf2abd42d90f34baf41
make -C connect_four_distillation/vendor/connect4 c4solver
.venv/bin/python -m unittest connect_four_distillation.test_oracle
```

The solver's AGPL-3.0 license applies to its source. The Connect Four
Transformer does **not** require this solver just to play.

## Restore Othello research checkout

From `start/`:

```bash
git clone https://github.com/likenneth/othello_world.git othello_world
git -C othello_world checkout f23bb5696cf30b93bd8af8a391ee33fc3aac417e
```

**Important:** the current local `othello_world/` also has uncommitted
Hyde-specific `lab/`, `config/`, `docs/`, and `scripts/` work, plus edits to
its own README and `.gitignore`. Cloning the pinned upstream commit does **not**
restore those local additions or the locally downloaded OthelloGPT checkpoint.
Keep/back up that checkout separately before moving machines. Its lab commands
in `start/README.md` are available in this original workspace, not in a clean
clone of this repository.

## Restore research references

From `start/game_transformer_knowledge_corpus/`:

```bash
../.venv/bin/python tools/fetch_sources.py
../.venv/bin/python tools/build_index.py
```

The downloader uses GitHub and arXiv and can take time and disk space. It
skips repository blobs over 8 MiB; the existing `download_manifest.json`
records revisions and omissions. `archives/` is an optional historical
download cache and is not needed by the source fetcher. The five-page MAV
takeaway PDF is also local-only; its [written takeaway](../start/game_transformer_knowledge_corpus/notes/06_mav_takeaway.md)
and the [paper](https://arxiv.org/abs/2412.12119) remain available.

## Regenerate game data only when needed

Check [Connect Four's dataset plan](../start/connect_four_transformer/DATASET_PLAN.md)
and [training strategy](../start/connect_four_transformer/STRATEGY_V5_UNIQUE_5MIN.md)
before rerunning generators. This is deliberately not part of environment
setup: generation and training consume CPU time.
