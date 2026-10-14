# Game Transformer Knowledge Corpus

An offline-first study kit for building a CPU-conscious Transformer that chooses moves in board games. Git keeps our source-guided notes, source catalog, and machine-readable index; downloaded reference repositories and PDFs are optional local resources, not part of the Hyde repository.

For the exact repository-and-paper shortlist requested, open [`REQUESTED_SET.md`](REQUESTED_SET.md).

## Start here

1. Read [`notes/00_core_idea.md`](notes/00_core_idea.md) for the complete mental model.
2. Read [`notes/01_repository_guide.md`](notes/01_repository_guide.md) before opening the nine codebases.
3. Follow [`notes/02_study_roadmap.md`](notes/02_study_roadmap.md) in order.
4. Use [`notes/03_cpu_first_architecture.md`](notes/03_cpu_first_architecture.md) as our implementation target.
5. Use [`notes/04_experiments_and_metrics.md`](notes/04_experiments_and_metrics.md) to decide whether a change actually helps.
6. Check [`docs/checkpoint_catalog.md`](docs/checkpoint_catalog.md) before downloading any weights.
7. Read the [MAV Transformer written takeaway](notes/06_mav_takeaway.md) before the [full paper](https://arxiv.org/abs/2412.12119). A five-page local PDF also exists in `papers/` on the original machine but is not committed.

## What is included

```text
game_transformer_knowledge_corpus/
├── README.md                 this map
├── repos/                    optional downloaded reference implementations
├── papers/                   optional downloaded primary PDFs
├── notes/                    our connected, source-guided explanation
├── rag/                      searchable JSONL chunks and manifest
├── tools/                    reproducible downloader and index builder
├── archives/                 optional local GitHub download cache
├── sources.json              canonical source catalog
└── download_manifest.json    exact revisions and intentionally skipped blobs
```

The current machine has downloaded snapshots, but these folders are ignored by the Hyde Git repository. After a fresh clone, run the fetcher below to recreate `repos/` and `papers/`. It downloads from current upstream branches; compare the result with `download_manifest.json` if an exact historical revision matters. External datasets and model checkpoints linked by those projects are **not** downloaded automatically: they can be huge, may have separate terms, and should be selected deliberately.

## The single idea connecting everything

A game-playing Transformer can be split into five replaceable pieces:

```text
game state/history → tokens → Transformer → policy/value → legal mask/search → move
```

- **OthelloGPT** learns next-move prediction from sequences and is ideal for understanding causal attention and learned board representations.
- **Laya/Jev-style decisions** show the useful interface: describe a state, enumerate candidate actions, and score the candidates in one pass.
- **Chess repositories** demonstrate richer tokenizers, policy/value heads, imitation learning, search, evaluation, and pretrained checkpoints.
- **Mamba/Jamba comparisons** test whether attention is necessary for the sequence model.
- **The CPU track** asks which representation, width, depth, batching, quantization, cache, and search budget give the most strength per millisecond.

## Recommended first experiment

Build a tiny candidate-action model for Tic-Tac-Toe, then swap only the game adapter to Othello:

```text
state tokens + candidate move tokens
              ↓
       4 blocks, width 128,
       4 heads, MLP width 512
              ↓
 policy logits over candidate moves + scalar value
              ↓
 legal-action mask → softmax → chosen move
```

Train first by imitation from an exact solver, evaluate against random/heuristic/minimax baselines, then add shallow search. This isolates the Transformer before game size and systems optimization complicate the experiment.

## Rebuild or search the corpus

```bash
../.venv/bin/python tools/fetch_sources.py
../.venv/bin/python tools/build_index.py

# Examples
rg -n "legal mask|masked softmax" notes repos
rg -n "class.*Attention|MultiheadAttention" repos
python3 -c 'import json; print(sum(1 for _ in open("rag/chunks.jsonl")))'
```

`rag/chunks.jsonl` is a checked-in snapshot that can be imported into a local retrieval system or searched directly with `rg`. Every chunk contains its source path and heading; rebuild it after refreshing the optional references. See the [external-resource guide](../../docs/EXTERNAL_RESOURCES.md).

## Boundaries

- A model that predicts **legal** moves is not automatically a model that chooses **winning** moves.
- Training loss alone is not playing strength; track win rate, Elo, illegal-move rate, value calibration, and latency.
- Search can make the same network much stronger but also changes the CPU budget.
- Jev is a hosted product; Laya is the open implementation in this corpus. Treat their interfaces as inspiration, not proof that their internals are identical.
- Preserve each repository's license and attribution when reusing code.

Retrieved on 2026-10-13. See [`sources.json`](sources.json) for source URLs and [`download_manifest.json`](download_manifest.json) for pinned commits.
