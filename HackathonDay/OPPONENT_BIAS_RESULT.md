# Opponent-bias experiment: result

The branch `experiment/opponent-bias` is an isolated worktree at
`/Users/anixd/Documents/Hyde-opponent-bias`. It copies the unchanged OneVOne
diagnostic agent and tests legal responses to observed mistakes. Neither this
copy nor its pre-event ancestor is a competition-ready submission under the
event's build-during-event rule.

## Evidence that motivated the candidate

The existing 200-game engine-side trace against official F found **13 saved
late positions with a deterministic winning alternative** to OneVOne's chosen
END_TURN or BUY_DEVELOPMENT_CARD. A separate check found **9 of 33 Monopoly
plays** had a resource choice below another legal choice under exact successor
scoring; five selected a resource the opponent held zero of. The live
candidate found all 13 previously identified winning lines within its CPU
search budget in an offline replay. Those positions were used for diagnosis,
so they cannot serve as a fresh strength test.

## Official-harness game tests

Kit commit `04fffac90e00610f60e0b06d919f262f4b3ed9db`; two-player 10-VP
rules, legal observation/index interface, both seats per board, enforced CPU
clock. The F comparison uses the **same** 100 boards, seeds
`2026101800`–`2026101899`, as the frozen 127/200 OneVOne baseline. Each row
changes only the listed candidate behavior. All tests had **zero illegal
indices, exceptions, timeouts, crashes, and clock overruns**.

| Candidate | Wins vs F / 200 | Matched games gained / lost vs OneVOne | Mean CPU/game |
| --- | ---: | ---: | ---: |
| Frozen OneVOne control | 127 | — | 111 ms |
| Guaranteed-win search + exact Monopoly | 127 | 0 / 0 | 118 ms |
| Plus two-trade → point-build planning | 126 | 0 / 1 | 120 ms |
| Instead remove buy bonus at 9 VP | 125 | 1 / 3 | 113 ms |
| Instead double opponent-production penalty | 127 | 0 / 0 | 119 ms |

On **fresh head-to-head boards**, seeds `2026110000`–`2026110099`, the
guaranteed-win + exact-Monopoly candidate won **101/200 (50.5%)** against the
unchanged OneVOne copy. The official evaluator's per-game 95% interval is
50.5% ± 6.9 percentage points; paired-board uncertainty may differ.

The saved [candidate vs F](reports/opponent_bias/candidate_vs_f_dev200.json),
[candidate vs OneVOne](reports/opponent_bias/candidate_vs_onevone_200.json),
[two-trade](reports/opponent_bias/two_trade_vs_f_dev200.json),
[9-VP buy](reports/opponent_bias/no_nine_vp_bonus_vs_f_dev200.json), and
[production-denial](reports/opponent_bias/deny_production_vs_f_dev200.json)
reports contain full game records, seed schedules, agent entry-point hashes,
and fault/CPU counters. The branch commits preserve each tested code version.

## Decision

**No candidate is promoted.** The specific late mistakes were real, but fixing
them did not increase wins against F on the development boards or establish an
advantage against OneVOne head to head. O is unpublished and was not tested.
The final worktree keeps only the bounded-win and exact-Monopoly changes as a
diagnostic reference. A future candidate needs a fresh paired game gain and
clock-safe confirmation before it is considered stronger.
