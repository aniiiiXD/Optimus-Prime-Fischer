# Isolated main-turn and discard rescues

[`phase_rescue_eval.py`](phase_rescue_eval.py) compares four arms on the same Catanatron v5 board seeds, seats, and mixed-opponent assignments: the frozen v3 fine-tuned Transformer, the frozen FairValue incumbent, FairValue in every `PLAY_TURN` choice only, and FairValue in every `DISCARD` choice only. All other decisions in each rescue arm come from the same Transformer. The runner is separate from the original five-arm [selective rescue](SELECTIVE_RESCUE.md) experiment and writes to a different output directory.

The [two intervention configs](configs/phase_rescue_play_turn.json) hash-pin the model checkpoint (`29b0f7418a873f34a02157dbde10b7f28f4d47ef2356d74076b55173efef3bbd`) and FairValue config (`5f908a453c87556463170378ef92d4ce79f03ecfdf0f1c304daae93fd3222ab3`). The runner rejects every board seed in the merged dataset's train, validation, or sealed final partitions. Choices are checked against the legal-action list; games have time, turn, and action caps.

Run the fresh eight-board diagnostic from the workspace root:

```bash
start/.venv/bin/python catanLabs/Competitive/phase_rescue_eval.py \
  --boards 8 --seed 20273600 --max-game-seconds 20 \
  --output-dir catanLabs/Competitive/runs/phase_rescue_8
```

Each completed game atomically updates `partial/<mix>__<arm>.json`. These files stay marked `unverified_partial` even after 4 × boards attempts; final arm reports and paired `*_vs_model.json` reports appear only after the full matched board, seat, color, and opponent grid is verified. Each report covers `mixed_diverse` (Weighted, Value, AlphaBeta1) and `mixed_strong` (Value, AlphaBeta1, AlphaBeta1). Rescue reports include consulted, disagreed, and override counts by phase. This is a diagnostic intervention, not a promotion test.

Focused tests:

```bash
start/.venv/bin/python -m unittest catanLabs.Competitive.tests.test_phase_rescue_agent -v
```

One-board smoke uses seed `20272952` in [`runs/phase_rescue_smoke`](runs/phase_rescue_smoke). It is for verifying the report path and legal play; four-game cells cannot establish playing strength.

## Fresh eight-board result, 16 October 2026

The [completed diagnostic](runs/phase_rescue_8/model.json) used seeds
**20273600–20273607**, all seats and both mixed opponent tables. All 256
games completed with zero caps or errors; matched assignments and partial
report integrity checks passed. The model and FairValue artifacts were
unchanged from the earlier selective-rescue screen.

| Arm | Diverse wins / 32 | Diverse mean VP | Strong wins / 32 | Strong mean VP |
|---|---:|---:|---:|---:|
| Frozen fine-tuned Transformer | 1 | 4.56 | 0 | 4.44 |
| FairValue control | 14 | 7.84 | 11 | 7.13 |
| Fair main-turn choices only | 8 | 5.84 | 7 | 5.78 |
| Fair discard choices only | 7 | 5.66 | 1 | 4.66 |

Main-turn rescue changed 251/356 actions (diverse/strong) and added
**+1.28/+1.34 mean final VP** over the frozen model. Its paired win gain was
+21.9 points in both mixes, with board-clustered 95% intervals +6.2 to
+37.5 and +9.4 to +40.6. Discard rescue changed 290/255 actions, adding
+1.09/+0.22 VP; its strong-table win gain was only +3.1 points. The
[main-turn comparison](runs/phase_rescue_8/play_turn_vs_model.json) and
[discard comparison](runs/phase_rescue_8/discard_vs_model.json) retain paired
outcomes and uncertainty.

This is a diagnostic, not a promotion: even after FairValue controls the
main turn, the hybrid trails the full FairValue control by roughly 1.3–2.0 VP.
Together with the [opening-only rescue](SELECTIVE_RESCUE.md), these results
show that the Transformer has important errors in several phases. They do
not establish that copying every FairValue decision with another broad
training run will beat FairValue.
