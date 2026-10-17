# Policy-guided search experiment

We tested the main idea behind several strong public Catan agents: let a fast
policy rank legal moves, then spend extra computation on promising sequences.
This experiment used our existing fair CPU value policy as the ranking and
leaf evaluator. Its search looks one more action ahead for roads, bank trades,
resource-granting cards, and settlement placement. It estimates development
purchases without seeing the next card and never samples the real next roll.
The implementation is [`guided_search_agent.py`](guided_search_agent.py), with
fixed parameters in [`guided_search_3.json`](configs/guided_search_3.json).

This is a **two-action lookahead**, not full MCTS or a new RL training run.
The inherited value policy remains the opponent-independent yardstick. We
also tested a hybrid that uses the first trained Transformer to nominate two
search roots per action type, alongside one root from the value policy.

## Matched four-player results

All games used the corrected v5 Catanatron tournament: four players, 10 VP,
no player-to-player offers, all four candidate seats, fixed `PYTHONHASHSEED`,
canonical legal-action order, isolated copy RNG, and forced chance outcomes
in simulated branches. The old fair policy played the **same boards and seats**
as the new agent.

| Opponents | New search, first 24 boards | Old fair policy | New search, 24 fresh confirmation boards | Old fair policy |
|---|---:|---:|---:|---:|
| Random | 96/96 | 96/96 | — | — |
| Weighted | 95/96 | 95/96 | — | — |
| Value | 33/96 | 34/96 | **39/96** | 29/96 |
| AlphaBeta1 | **41/96** | 32/96 | **47/96** | 32/96 |

The first set used seeds **20262200–20262223**; the confirmation used
**20262300–20262323**. Every candidate and control game completed with no
illegal action or agent error. Pooled over 48 independent boards with all
four seats rotated:

| Opponents | Search | Old fair policy | Paired win-rate difference | Board-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Value | 72/192 (37.5%) | 63/192 (32.8%) | +4.7 points | −3.1 to +12.0 points |
| AlphaBeta1 | **88/192 (45.8%)** | 64/192 (33.3%) | **+12.5 points** | **+5.7 to +19.3 points** |

These are board-clustered, 10,000-resample paired intervals. Matching board
and seat does not force identical later dice after agents choose different
actions. Search's AlphaBeta1 gain is supported by both independent seed sets.
The Value gain is still uncertain when pooled.

## Mixed opponents and learned prior

On 24 additional boards, seeds **20262400–20262423**, we rotated opponent
placement as well as the candidate seat:

| Other players | Search | Old fair policy | Paired difference, 95% interval |
|---|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | **59/96 (61.5%)** | 50/96 (52.1%) | +9.4 points, 0.0 to +17.7 |
| Value + AlphaBeta1 + AlphaBeta1 | 36/96 (37.5%) | **38/96 (39.6%)** | −2.1 points, −9.4 to +5.2 |

The diverse-mix gain and stronger-mix loss both have intervals containing
zero. This limits a broad promotion claim, despite the convincing uniform
AlphaBeta1 result.

The **Transformer-prior hybrid** used the existing 77,569-parameter pilot
checkpoint, with no new training. On the first 24-board set it won **35/96
against Value** and **40/96 against AlphaBeta1**, compared with the fair
search's 33/96 and 41/96. Paired differences between these search variants
were uncertain (Value +2.1 points, 95% −5.2 to +9.4; AlphaBeta1 −1.0 point,
95% −8.3 to +6.2). The learned prior therefore has **no demonstrated playing
strength benefit** yet. The Transformer hybrid took about 1.15–1.23 ms at
the median decision, versus 0.22 ms for fair search; p95 was about 9.3 ms
versus 7.7–7.8 ms on the same strong-opponent boards.

The fair search passed a hidden-state perturbation check on **172 decisions**;
the Transformer hybrid passed **163**. Altering opponents' hidden resource
types, development-card order, and RNG left their choices unchanged and did
not advance the live game's RNG. This is a regression check, not a formal
restricted-observation API.

## Decision and artifacts

The fair-policy search is a strong **AlphaBeta1 specialist** and the best
uniform-AlphaBeta1 performer in our local v5 test. The old fair policy stays
the general incumbent because search did not beat it in the stronger mixed
table. We need a wider opponent mix or a changed search design before
promoting it as a universal Catan policy. The Transformer prior is retained as
an ablation, not a strength upgrade. No neural weights were trained here;
this used **zero of the 10-minute CPU training allowance**.

- First four-tier reports: [`search`](runs/guided_search_holdout_v5.json),
  [`control`](runs/guided_search_control_v5.json),
  [`comparison`](runs/guided_search_comparison_v5.json)
- Confirmation reports: [`search`](runs/guided_search_confirm_v5.json),
  [`control`](runs/guided_search_confirm_control_v5.json),
  [`comparison`](runs/guided_search_confirm_comparison_v5.json)
- Pooled 48-board comparison:
  [`guided_search_pooled_v5.json`](runs/guided_search_pooled_v5.json)
- Mixed tables: [`search`](runs/guided_search_mixed_v3.json),
  [`control`](runs/guided_search_mixed_control_v3.json),
  [`comparison`](runs/guided_search_mixed_comparison_v3.json)
- Transformer hybrid:
  [`configuration`](configs/transformer_guided_search.json),
  [`tournament`](runs/transformer_guided_holdout_v5.json),
  [`versus fair search`](runs/transformer_vs_fair_guided_v5.json)
- Repeat the visibility regression with
  `PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/check_visibility.py --agent guided --config catanLabs/Competitive/configs/guided_search_3.json`.

The benchmark is local Catanatron play, not a public or human tournament.
It uses three identical opponents in each uniform tier and excludes domestic
trade offers. Search assumes the runner's corrected copy/chance behavior.
