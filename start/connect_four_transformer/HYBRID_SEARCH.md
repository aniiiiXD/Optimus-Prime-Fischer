# Experimental Transformer-guided alpha-beta

See [COMPETITION_READINESS.md](COMPETITION_READINESS.md) for the selected CPU
configuration, held-out results, and limits of the current claim.

This is a separate agent; it does not change `TransformerAgent`, the existing
`AlphaBetaTeacher`, or any saved checkpoint. It uses the current Transformer
for root move ordering, while game rules and adversarial search decide the move.
Learned leaf values are optional and disabled by default.

## Play

From the workspace's `start/` directory:

```bash
.venv/bin/python -m connect_four_transformer.play_hybrid \
  --checkpoint connect_four_distillation/artifacts/checkpoints/candidate_pons_phases_v3_5min.pt \
  --human X --depth 5 --nodes 5000 --model-evals 1 --model-value-weight 0
```

Human columns are entered as 1–7. Search actions and the Python API use
zero-based columns 0–6. The default checkpoint is the Pons phase-v3 model.

## Python API

```python
from pathlib import Path
from connect_four_engine import ConnectFour
from connect_four_transformer.hybrid_search import TransformerGuidedSearchAgent

agent = TransformerGuidedSearchAgent.from_checkpoint(
    Path("connect_four_distillation/artifacts/checkpoints/candidate_pons_phases_v3_5min.pt"),
    max_depth=5,
    node_limit=5_000,
    model_value_weight=0.0,
    max_model_evaluations=1,
)
game = ConnectFour.from_moves([3, 2, 4])
result = agent.evaluate(game)
print(result.best_action, result.reason, result.stats.to_dict())
```

`choose(game)` returns only a legal action and can be used with the existing
tournament-agent protocol. `evaluate(game)` returns:

- `best_action`: selected zero-based column;
- `model_state_value`: the checkpoint's root value estimate, not the searched
  minimax score;
- `action_scores`: searched horizon scores for root actions. Scores for
  non-principal moves may be alpha-beta bounds, not exact minimax values;
- `reason`: `take_immediate_win`, `forced_one_ply_block`,
  `alpha_beta_search`, or `transformer_budget_fallback`;
- `stats`: visited nodes, completed iterative-deepening depth, cutoffs/cache
  hits, neural calls, and elapsed wall time.

## Decision recipe

1. The rules engine finds an immediate win and takes it regardless of the
   model's opinion.
2. If the opponent has an immediate winning drop, try every legal response;
   when any response removes all immediate replies, search only those safe
   blocks. This is a sound one-ply forced-block filter.
3. Otherwise, use the model's policy probability plus its action-Q estimate to
   order root branches. Center-first order breaks ties.
4. Iterative-deepening negamax alpha-beta explores adversarial replies. At
   horizon leaves, an immediate win or unavoidable next-turn loss takes
   precedence; remaining leaves use the handcrafted window/center heuristic by
   default. A nonzero `model_value_weight` enables learned state values.
5. The node budget bounds tree work, and the separate model-call cap bounds
   CPU inference. If an iteration runs out of nodes, only the last fully
   completed depth is used; if depth one does not finish, it falls back to
   the legal model ranking.

The checkpoint is not retrained or modified. This is search-time combination,
not RL, self-play, or a new distilled model.

## Limits / interpretation

- Depth is a fixed-ply horizon, not proof of a forced win. The window heuristic
  and learned value are approximate past the explicit immediate tactics.
- The current model has modest teacher accuracy; model values can mislead leaf
  evaluation. `model_value_weight=0` is the measured setting; raise the value
  weight and model-call cap only as a separately benchmarked experiment.
- Root search scores can be bounds due to alpha-beta pruning, so do not read
  them as calibrated probabilities or exact win chances.
- Win rates need paired openings, both colors, and more than one seed before
  claiming a real improvement. No tournament-readiness claim is made here.

## Focused regression tests

```bash
.venv/bin/python -m unittest connect_four_transformer.test_hybrid_search -v
```

The tests include an intentionally left-biased fake model to verify that exact
immediate wins and forced blocks override a bad neural preference, and that
budget fallback preserves the caller's board state.

## Strength and CPU benchmark

Run the paired comparison from `start/`:

```bash
.venv/bin/python -m connect_four_transformer.benchmark_hybrid_search \
  --games-per-side 15 --seeds 43027,43028 \
  --report connect_four_transformer/artifacts/reports/hybrid_search.json
```

Every opening is replayed with the candidate as both X and O. The saved JSON
contains per-game outcomes, paired-opening bootstrap intervals, and move
latency/search counters for plain guarded Transformer, plain depth-5 alpha-beta,
and several Transformer-guided search budgets. `hybrid_d5_no_model_value` keeps
the Transformer for root move ordering but uses only the hand-built leaf value;
it intentionally avoids redundant Transformer calls at search leaves. Treat
the current depth-3 opponent as an internal baseline, not a top external engine.
