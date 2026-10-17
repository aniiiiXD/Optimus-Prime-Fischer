# Last-seat joint opening planner: bounded shadow probe

**Status: unpromoted research candidate.** [`opening_pair_agent.py`](opening_pair_agent.py)
plans the fourth player's consecutive settlement → road → settlement → road
opening. It scores public tile production, usable resource combinations,
second-settlement starting cards, modest port value, and nearby expansion
sites. It uses FairValue for every other seat and phase. The first version
searches at most 12 first settlements, 4 second settlements per first road,
and 432 complete lines; it validates the selected four-action line against
the engine's legal actions before storing it. Each stored continuation is
rechecked against live legal actions when played.

The [four-board shadow report](runs/opening_pair_shadow_4boards.json) used
fresh seeds **20272700–20272703**, provisional Catanatron v5 rules, and
Value + two AlphaBeta1 opponents during the initial setup. FairValue made
every actual move. The candidate was scored only at the fourth seat's first
placement, and its full line was replayed legally in a copied game.

| Seed | Candidate first action differs | Candidate pair proxy | FairValue pair proxy | Scan time |
| --- | --- | ---: | ---: | ---: |
| 20272700 | Yes | 28.047 | 24.475 | 19.777 ms |
| 20272701 | Yes | 28.167 | 24.877 | 19.447 ms |
| 20272702 | No | 28.147 | 28.147 | 18.693 ms |
| 20272703 | Yes | 28.738 | 27.032 | 18.845 ms |

The planner changed three of four full opening lines. Median scan latency
was **19.146 ms**, p95 **19.728 ms** on these four samples. A second
[16-board shadow](runs/opening_pair_shadow_16boards.json) on seeds
20273100–20273115 changed 12/16 first placements and 16/16 full lines;
median/p95/max planning time was 19.069/20.017/20.342 ms. Every plan passed
legal replay; hidden deck/RNG perturbation left all four plans unchanged;
the 16-board probe also checked legal replay and hidden-state/RNG invariance
for every board. Scoring did not advance the live RNG. The pair score is the candidate's own
heuristic, so higher values are expected and **are not evidence of stronger
gameplay**. See the separate [matched game screen](OPENING_PAIR_SCREEN.md)
for playing-strength evidence.

Focused tests in [`test_opening_pair_agent.py`](tests/test_opening_pair_agent.py)
cover four legal continuations, search caps, hidden-state/RNG invariance,
FairValue fallback outside the eligible opening, stale-plan fallback, and
configuration validation. They passed 4/4 in 0.404 seconds.

Reproduce:

```bash
start/.venv/bin/python -m unittest catanLabs.Competitive.tests.test_opening_pair_agent -v
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/opening_pair_shadow_probe.py --boards 4 --seed 20272700 --report catanLabs/Competitive/runs/opening_pair_shadow_4boards.json
```

The fresh matched all-seat screen has been run; its small gain does not
justify a 100-board confirmation for this version. Other seats intentionally
remain FairValue. Do not promote from proxy-score or shadow results.
