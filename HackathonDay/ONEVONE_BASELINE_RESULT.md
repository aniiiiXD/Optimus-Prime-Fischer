# OneVOne: first official-harness baseline

The self-contained copy won **127/200 games (63.5%) against official F**.
No neural training or coefficient tuning was performed.

| Run | Games / boards | Wins | Mean CPU/game | Worst CPU/game | Budget overruns | Errors | Timeouts / crashes |
|---|---:|---:|---:|---:|---:|---:|---:|
| Smoke | 20 / 10 | 13/20 (65.0%) | 97.2 ms | 246.2 ms | 0 | 0 | 0 / 0 |
| Initial baseline | 200 / 100 | 127/200 (63.5%) | 111.4 ms | 333.3 ms | 0 | 0 | 0 / 0 |
| Fresh confirmation | 1,000 / 500 | 609/1,000 (60.9%) | 118.8 ms | 494.3 ms | 0 | 0 | 0 / 0 |

The evaluator reports a 95% interval of **63.5% ± 6.67 percentage points**
(approximately 56.8%–70.2%). This is the official script's interval; its
formula treats individual games as independent, although boards are paired.
It is not a board-clustered uncertainty estimate.

The fresh 1,000-game run used seeds **2026200000–2026200499**, each board
played from both seats. The evaluator reported a 95% interval of **60.9% ±
3.02 percentage points**, zero errors in 50,753 decisions, and a 251.2-second
wall time. It is a local result against F, not against the organizers' private
O bot. Its CPU times are measured on the local Mac, not the event machine.

## What was adapted

- Copied the OneVOne policy, three helper modules and default configuration.
- Replaced imports from the wider workspace with imports inside the agent folder.
- Added `Agent.decide(obs, playable_actions)`, a permitted sampled world, and
  action-to-index conversion.
- Made configuration loading explicit and local, and corrected rule metadata
  to disable player trades. This metadata does not alter the official engine.
- Preserved every scoring coefficient and tactical-search limit.

## Protocol

- Official kit: `04fffac90e00610f60e0b06d919f262f4b3ed9db`.
- Official engine: `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`.
- Two players, 10 VP; official masking, both seats, CPU clock enforced.
- Opponent: the kit's `examples/f_agent.py`, not a depth-two web bot.
- Smoke seeds: 2026101700–2026101709.
- Baseline seeds: 2026101800–2026101899; disjoint from the smoke boards.
- Python 3.11.15 on local macOS arm64; pinned numpy/networkx/psutil.
- Unused torch/scipy/scikit-learn were omitted. The heuristic imports none of
  them; downloading the unused ML wheels was stopped before installation.
- Official self-check passed, including the reference game and replay checks.

Timing is local development-machine CPU. The organizer machine remains the
reference for scored timing and enforces its own resource isolation. No result
against organizer bot O is available from these runs.

## Interpretation

The adapter works and the original policy fits the official clock in this
screen. The baseline is 63.5% against F on these 100 boards. This is not a
measured improvement over a previous agent, and it does not establish strength
against O or other teams.

Next: preserve the baseline, record and inspect losing-game decisions, then
compare one change at a time on identical paired boards and confirm the chosen
change on unused boards. There is no observed clock failure here to justify
pruning the baseline solely on timing grounds.

## Reproduce and inspect

Run from the Hyde root after restoring the pinned official kit and environment:

```sh
HackathonDay/.venv/bin/python HackathonDay/evaluate_baseline.py --selfcheck --label selfcheck_repeat
HackathonDay/.venv/bin/python HackathonDay/evaluate_baseline.py --games 20 --seed0 2026101700 --label smoke_repeat
HackathonDay/.venv/bin/python HackathonDay/evaluate_baseline.py --games 200 --seed0 2026101800 --label baseline_repeat
```

The recording driver launches the unmodified `catan_kit.evaluate` CLI and
retains its full output, command, source hashes and environment metadata.

- [Self-check output](reports/onevone_baseline/selfcheck.txt)
- [Smoke output](reports/onevone_baseline/smoke_20.txt) and [manifest](reports/onevone_baseline/smoke_20.json)
- [200-game output](reports/onevone_baseline/baseline_200.txt) and [manifest](reports/onevone_baseline/baseline_200.json)
- [Fresh 1,000-game output](reports/onevone_baseline/baseline_f_1000_fresh.txt) and [manifest](reports/onevone_baseline/baseline_f_1000_fresh.json)
- [Local dependency versions](reports/onevone_baseline/environment.freeze.txt)
- [Copy provenance](onevone_baseline/PROVENANCE.json)
- [Adaptation plan and submission eligibility](ONEVONE_ADAPTATION_PLAN.md)
