# Competitive Catan policy: CPU experiment

**Selected agent:** [`fair_value_agent.py`](fair_value_agent.py) with
[`fair_army_finish.json`](configs/fair_army_finish.json). This is a CPU strategy
policy that can serve as a stronger teacher for future Transformer work. It
requires no neural weights at inference time.

## Corrected tournament

We evaluated **24 new board seeds, 20261900–20261923, with all four seats
rotated**. Each game had four players, 10 points to win, no player-to-player
trade offers, and three identical opponents from its tier. The original
Catanatron Value bot ran on the same board and seat grid. Protocol v5 isolates
RNG in hypothetical game copies, correctly forces dice/card outcomes in search
branches, fixes Python's hash seed, and sorts legal actions for repeatable
tie-breaking. Two independent same-seed smoke runs produced identical game
records. Earlier v1–v4 reports are exploratory and are not mixed with these
results.

| Three opponents | Selected fair policy | Original Value candidate | Selected mean VP | Decision median / p95 |
|---|---:|---:|---:|---:|
| Random | **96/96 (100%)** | 96/96 (100%) | 10.15 | 0.454 / 5.024 ms |
| Weighted | **95/96 (99.0%)** | 95/96 (99.0%) | 10.09 | 0.459 / 5.343 ms |
| Value | **32/96 (33.3%)** | 24/96 (25.0%) | 7.10 | 0.412 / 4.749 ms |
| AlphaBeta1 | **28/96 (29.2%)** | 27/96 (28.1%) | 7.08 | 0.388 / 4.088 ms |

All 384 games for the selected agent completed without an illegal action or
agent error. Its Wilson 95% win-rate ranges were 24.7–43.2% against Value
and 21.0–38.9% against AlphaBeta1. On this test it beat three Value bots more
often than a Value bot did, and played above the four-player equal-seat share
against three AlphaBeta1 bots. The AlphaBeta1 edge is still statistically
uncertain.

Paired by board seed and seat, the win-rate difference over the original
Value candidate was +8.3 percentage points against Value and +1.0 point
against AlphaBeta1. A 10,000-resample bootstrap over board seeds gave 95%
intervals of 2.1 to 14.6 points and −9.4 to 10.4 points. A matched board does
not force identical future dice after different policies take different
actions; this controls layout and seat, not every game event.

We also tested two mixed opponent compositions on **24 further seeds,
20261950–20261973**, again rotating all four seats and rotating which rival
sat in each opponent position:

| Other three players | Selected fair policy | Original Value candidate | Selected mean VP |
|---|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | **38/96 (39.6%)** | 34/96 (35.4%) | 7.37 |
| Value + AlphaBeta1 + AlphaBeta1 | **31/96 (32.3%)** | 25/96 (26.0%) | 7.06 |

All 192 mixed games completed without illegal actions or agent errors. The
board-clustered difference intervals were −4.2 to 12.5 percentage points for
the diverse mix and −2.1 to 14.6 points for the stronger mix. These are useful
generality checks; neither mix shows a statistically clear edge over Value at
this sample size.

## How we reached this policy

The first Transformer won 0/16 against Value and 0/16 against AlphaBeta1 in
its historical pilot. We tried direct production/expansion scoring, altered
Value weights, and depth-two search. The useful policy retained Catanatron's
point and production value function, increased rewards for connected
settlement sites and hand combinations, valued played Knights, and encouraged
development purchases from seven points onward.

The main policy handles chance and hidden information explicitly:

- **Roll:** average value over all 11 dice sums using their probabilities.
- **Development purchase:** value an unknown card count and deduct the spent
  resources without inspecting the deck.
- **Monopoly:** estimate resources from public production.
- **Robber:** score blocked enemy production, leader threat, and own loss using
  public board and scores.

On a separate historical v3 AlphaBeta1 screen of 12 boards (48 games), the selected
setting won 12; omitting its late card bonus won 11, omitting its extra army
value won 9, and omitting both won 10. A public-board Value robber variant
won 13. Those differences are too small to establish which of these nearby
settings is best, so the selected agent stays the one tested across the full
four-tier tournament.

The selected policy passed a visibility perturbation check on **more than 100
decisions**. Changing hidden opponent resources, deck order, and RNG did not
change its action or advance the live game's RNG. The check included a
synthetic state with Roll, Knight, and Monopoly available. It is a useful
regression check, not a formal guarantee about every state.

## Limits and artifacts

The upstream Catanatron opponents still receive the simulator's full game
object. The selected policy uses only public board/score data, its own private
hand/cards, and legal actions, but the harness does not enforce this as a
restricted observation type. These scores apply to this four-player
Catanatron setup; trading with other players is excluded. A public Catan
leaderboard or human-level claim would need a different evaluation.

- Agent and selected parameters: [`fair_value_agent.py`](fair_value_agent.py),
  [`fair_army_finish.json`](configs/fair_army_finish.json)
- Final four-tier JSON: [`fair_candidate_v5.json`](runs/fair_candidate_v5.json)
- Original Value control: [`value_baseline_v5.json`](runs/value_baseline_v5.json)
- Paired comparison: [`fair_vs_value_v5.json`](runs/fair_vs_value_v5.json)
- Mixed opponent results: [`mixed_candidate_v3.json`](runs/mixed_candidate_v3.json),
  [`mixed_value_v3.json`](runs/mixed_value_v3.json), and
  [`mixed_comparison_v3.json`](runs/mixed_comparison_v3.json)
- Visibility check: [`check_visibility.py`](check_visibility.py)
- Reproduction command: [`README.md`](README.md)

The earlier 10-minute CPU Transformer run is unchanged. This experiment
tuned a strategy's parameters through games and did not train neural weights.
