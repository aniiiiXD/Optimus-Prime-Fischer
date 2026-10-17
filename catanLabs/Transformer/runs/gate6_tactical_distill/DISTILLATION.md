# TacticalVP Transformer distillation: first controlled result

**Decision, 16 October 2026: do not promote the distilled Transformer.** A
visible-only Transformer was trained to imitate the promoted TacticalVP
teacher, then screened on fresh boards. It is fast and legal, but loses
substantial playing strength to the teacher.

## Data and CPU budget

- Teacher and trajectory policy: frozen visible-only `TacticalVPAgent`,
  [`tactical_vp_min0.json`](../../../Competitive/configs/tactical_vp_min0.json).
- Corpus: 30,929 unique player-visible decisions from 102 completed all-seat
  TacticalVP self-play games. The raw reconstruction audit sampled 64 rows,
  found zero duplicate, hidden-field, or metadata failures, and froze a
  board-disjoint train/validation/final split.
- Model: 77,569-parameter, two-layer width-64 action Transformer.
- CPU optimizer budget: 590 seconds, 4 threads; 24 epochs; best epoch 22.
  The held-out validation set contained 2,764 decisions from 10 untouched
  board seeds. Overall agreement was 81.0%; agreement where more than one
  action was legal was 68.0%.

## Fresh game screen

Candidate and TacticalVP control used the same new eight boards, all four
candidate seats, opponent rotations, and provisional Catanatron v5 rules.
All 128 games completed legally and without caps or errors.

| Opponents | Transformer | TacticalVP | Paired win difference | 95% interval | VP difference |
|---|---:|---:|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | 5/32 | 12/32 | −21.9 points | −34.4 to −9.4 | −1.406 |
| Value + two AlphaBeta1 | 5/32 | 10/32 | −15.6 points | −25.0 to −3.1 | −1.000 |

The Transformer had 1.16/1.20 ms all-decision p95 latency in the two tables,
versus TacticalVP's 6.61/7.37 ms. Speed does not compensate for the measured
strength loss.

## Why the overall imitation score misled us

The held-out phase table is the actionable diagnosis:

| Decision family | Agreement | What it means |
|---|---:|---|
| Opening settlement | 11/80 = 13.8% | The model does not yet choose the board-defining first placement. |
| Opening road | 42/80 = 52.5% | It often loses the intended opening continuation. |
| Robber placement | 21/129 = 16.3% | It does not preserve the teacher's public-board targeting behavior. |
| Road action | 90/314 = 28.7% | Board-specific expansion remains poorly represented. |
| Play-turn overall | 86.0% | Inflated by forced or easy end-turn decisions; nonforced play-turn agreement is 68.9%. |
| Discard | 94.6% | Well learned, but not the main source of strength. |

The teacher's exact tactical override is sparse and the model sees only its
selected action. Errors compound over hundreds of sequential moves, so the
headline action match cannot establish game strength.

Next experiment: use phase-balanced sampling or loss weights for opening,
robber, and board-building actions, report their held-out agreement separately,
and run the same fresh all-seat screen. Keep TacticalVP as incumbent unless a
candidate passes the established game-outcome gate.

Evidence: [dataset manifest](tactical_selfplay_100.json),
[audit](tactical_selfplay_100.audit.json), [training summary](transformer/training.json),
[phase diagnostics](transformer/diagnostics.json), [candidate screen](transformer/mixed_screen_8.json),
[TacticalVP control](tactical_control_mixed_screen_8.json), and
[paired result](transformer_vs_tactical_screen_8.json).

## Phase-balanced follow-up: rejected

The same frozen dataset, architecture, split, and maximum CPU budget were run with a 6× epoch sampling weight for `BUILD_INITIAL_SETTLEMENT`, `BUILD_INITIAL_ROAD`, and `MOVE_ROBBER` rows. The trainer stopped on a validation plateau after 468 CPU seconds at epoch 11.

It improved the most deficient offline phases only modestly: opening settlement **13.8%→25.0%** and robber **16.3%→20.2%**. It harmed the rest of the policy: overall agreement **81.0%→66.7%**, nonforced agreement **68.0%→43.7%**, and nonforced play-turn agreement **68.9%→53.9%**.

Fresh outcome evidence rejects this intervention. On seeds 20275100–107, the phase-weighted model won **2/32 versus 19/32** TacticalVP in the diverse mix (−53.1 points; 95% interval −71.9 to −31.2) and **3/32 versus 12/32** in the strong mix (−28.1 points; interval −40.6 to −15.6). All games completed legally with no caps or errors.

This rules out simple phase oversampling as the next route. The next ML change should preserve the exact TacticalVP decision path as a search or policy fallback while a model contributes a narrow, measurable component, such as a root ranking or action-type prior. Do not replace the incumbent with a raw behavioral clone.

Additional evidence: [weighted training](transformer_rare6/training.json), [weighted diagnostics](transformer_rare6/diagnostics.json), [weighted screen](transformer_rare6/mixed_screen_8.json), [matched control](tactical_control_rare6_mixed_screen_8.json), and [paired result](transformer_rare6_vs_tactical_screen_8.json).
