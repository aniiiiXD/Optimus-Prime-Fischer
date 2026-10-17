# Selective FairValue rescue of the frozen v3 fine-tune

[`selective_rescue_eval.py`](selective_rescue_eval.py) runs five candidates on the same fresh Catanatron v5 boards, seats, and mixed-opponent assignments:

| Arm | Decisions |
|---|---|
| `model` | Unmodified frozen v3 fine-tune. |
| `fair` | Unmodified FairValue incumbent. |
| `opening` | FairValue replaces model choices on initial settlement and road prompts. |
| `robber` | FairValue replaces model choices on robber prompts. |
| `trade_end` | FairValue replaces a main-turn model choice only when one policy chooses `MARITIME_TRADE` and the other chooses `END_TURN`. |

The wrapper scores the model on a player-visible snapshot and checks both choices against the legal-action list. It records candidate decisions, consulted fair decisions, disagreements, actual overrides, and the two trade/end directions for each game. The frozen checkpoint SHA-256 is `29b0f7418a873f34a02157dbde10b7f28f4d47ef2356d74076b55173efef3bbd`; the FairValue config SHA-256 is `5f908a453c87556463170378ef92d4ce79f03ecfdf0f1c304daae93fd3222ab3`. The three [configs](configs/selective_rescue_opening.json) pin these hashes and fail if either artifact changes. The runner refuses board seeds in the merged dataset's train, validation, or sealed final split.

Run the **fresh eight-board diagnostic** from the workspace root after review:

```bash
start/.venv/bin/python catanLabs/Competitive/selective_rescue_eval.py \
  --boards 8 --seed 20273500 --max-game-seconds 20 \
  --output-dir catanLabs/Competitive/runs/selective_rescue_8
```

The runner writes `model.json`, `fair.json`, three rescue reports, and paired `*_vs_model.json` comparisons. Each report has both `mixed_diverse` (Weighted, Value, AlphaBeta1) and `mixed_strong` (Value, AlphaBeta1, AlphaBeta1), with four seats per board and the usual legal-action and game-cap checks. Compare any two arm reports with [`compare.py`](compare.py); the shared model and fair hashes are part of its rules comparison. Use paired final VP, wins, and override counts to choose which decision class to investigate further. This is a causal policy intervention within the local v5 simulator, not a promotion test or proof that a teacher action is optimal.

After every game, the runner atomically updates `partial/<mix>__<arm>.json` with its recorded attempts, expected attempts, running summary, and game rows. These files are explicitly marked `unverified_partial`, including when an arm reaches its expected count. Final arm and paired reports are written only after every arm has a complete, identical board, seat, color, and opponent-assignment grid and the paired comparisons succeed. An interrupted run therefore preserves its completed games without presenting a partial tier as a verified comparison.

## One-board smoke

Seed `20272950`, four seats, both mixes, and a 20-second game cap were run once. All **40 games completed**, with no illegal actions or caps; every arm had the same candidate color and opponent-color assignments. Override counts were 10 and 12 for opening, 14 and 11 for robber, and 3 and 3 for trade/end (diverse and strong mixes, respectively). The unmodified model and fair control each won 1/4 in each mix. These four-game cells have no useful strength precision; the [smoke reports](runs/selective_rescue_smoke/model.json) only validate wiring, legal play, counters, and `compare.py` compatibility.

The partial-report path was also smoke-tested on seed `20272951`: all 40 games completed, all ten partial files recorded 4/4 attempts, all eight final and paired files were written, and no temporary write files remained. These results are plumbing checks only.

Focused wrapper, seed-gate, partial-report, and matched-grid tests: `start/.venv/bin/python -m unittest catanLabs.Competitive.tests.test_selective_fair_rescue_agent -v`.

## Fresh eight-board diagnostic, 16 October 2026

The [matched run](runs/selective_rescue_8/model.json) used seeds
**20273500–20273507**, all four seats, both mixed opponent tables, and a
20-second game cap. All 320 games completed without errors or caps. The
runner saved atomic per-game partial files, then final reports after checking
every arm's seed, seat, candidate color, and opponent assignment grid. The
focused tests passed 7/7 before this run.

| Arm | Diverse wins / 32 | Diverse mean VP | Strong wins / 32 | Strong mean VP |
|---|---:|---:|---:|---:|
| Frozen fine-tuned Transformer | 2 | 4.66 | 1 | 4.41 |
| FairValue control | 12 | 7.25 | 12 | 7.31 |
| Fair opening only | 5 | 5.94 | 2 | 5.44 |
| Fair robber only | 1 | 4.91 | 1 | 4.34 |
| Fair trade/end disagreement only | 2 | 4.91 | 0 | 4.47 |

Opening rescue changed 88/89 actions in the diverse/strong games and gained
**+1.28/+1.03 mean final VP** over the full model. Its paired win gains were
+9.4 points (board-bootstrap 95% interval −3.1 to +21.9) and +3.1 points
(0 to +9.4). Robber rescue changed 109 actions in each table; trade/end
rescue changed 51/69. Neither raised wins consistently. The saved
[opening](runs/selective_rescue_8/opening_vs_model.json),
[robber](runs/selective_rescue_8/robber_vs_model.json), and
[trade/end](runs/selective_rescue_8/trade_end_vs_model.json) comparisons carry
the paired game evidence and intervals.

**Interpretation:** bad opening choices account for a material share of the
frozen model's VP gap, but a fair opening does not close its roughly two-to-three
VP gap to FairValue. These are policy interventions on only eight boards, not
proof that every teacher opening is optimal or a model promotion. The next
diagnostic should isolate the remaining main-turn and discard decisions rather
than spend another CPU training budget on undifferentiated fair-action labels.
