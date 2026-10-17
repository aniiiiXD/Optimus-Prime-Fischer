# Guided search phase gate

**Decision: no promotion.** The new
[`phase_guided_agent.py`](phase_guided_agent.py) holds both frozen policies and
routes by Catanatron prompt. The opening arm uses guided search for the two
initial settlement/road prompts and FairValue elsewhere. The turn arm uses
guided search only in `PLAY_TURN`. The all arm uses guided search in every
nonforced phase. All arms load the same existing
[`guided_search_3.json`](configs/guided_search_3.json) and
[`fair_army_finish.json`](configs/fair_army_finish.json); neither policy was
edited. Forced actions pass through directly, and chosen actions are checked
against the supplied legal list.

This tests whether guided search's historical mixed-strong regression comes
from opening choices or main-turn choices. The earlier 24-board mixed run
reported **36/96 guided versus 38/96 FairValue** in the strong composition,
while guided improved the diverse composition **59/96 versus 50/96**. Its
strong seat breakdown was guided/Fair **13/13, 8/9, 3/6, 12/10** for seats
0–3; the losses were concentrated in seats 1–2. The diverse breakdown was
**14/14, 16/12, 16/11, 13/13**. These historical data motivate a phase
intervention but cannot identify its cause.

## Fresh matched 16-board screen

All four arms played seeds **20273800–20273815**, four seats per board, both
mixed compositions, and the same v5 rules and opponent assignments. All
**512 games completed** with zero caps or errors. Paired differences and 95%
intervals use board-clustered bootstrapping. The
[matched report](runs/phase_guided_compare_16.json) checks the four-seat grid,
rules, and opponent assignments.

| Opponents | Arm | Wins / 64 | Fair wins / 64 | Paired win-rate difference | 95% interval | Mean VP difference |
|---|---|---:|---:|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | Guided opening | 26 | 27 | −1.6 points | −14.1 to +10.9 | −0.27 |
| Weighted + Value + AlphaBeta1 | Guided turn | 34 | 27 | +10.9 points | −1.6 to +23.4 | +0.33 |
| Weighted + Value + AlphaBeta1 | Guided all | 30 | 27 | +4.7 points | −9.4 to +18.8 | −0.11 |
| Value + two AlphaBeta1 | Guided opening | 18 | 26 | **−12.5 points** | **−21.9 to −3.1** | −0.84 |
| Value + two AlphaBeta1 | Guided turn | 20 | 26 | −9.4 points | −20.3 to +3.1 | −0.02 |
| Value + two AlphaBeta1 | Guided all | 19 | 26 | −10.9 points | −23.4 to +1.6 | −0.38 |

Seat-level wins on the strong table (each cell is wins/16):

| Seat | Fair | Guided opening | Guided turn | Guided all |
|---:|---:|---:|---:|---:|
| 0 | 6 | 6 | 5 | 5 |
| 1 | 8 | 5 | 5 | 5 |
| 2 | 4 | 4 | 5 | 5 |
| 3 | 8 | 3 | 5 | 4 |

The opening-only intervention has a substantial measured cost against the
strong mix on these boards, especially seat 3. Turn-only also lost six wins
against that mix, but its interval includes zero and its mean VP was nearly
unchanged. The diverse table moves in the opposite direction for turn-only.
This suggests **both phases can matter by opponent mix**; it does not prove
that one phase alone explains the historical full-guided regression.

## One-board smoke, not a strength estimate

Fresh seed **20272800**, all four seats, both mixed opponent compositions,
provisional four-player v5 first-to-10 rules, and 20-second game caps:

| Composition | FairValue | Guided opening only | Guided turn only | Guided all |
|---|---:|---:|---:|---:|
| Weighted + Value + AlphaBeta1 | 1/4 | 1/4 | 3/4 | 3/4 |
| Value + two AlphaBeta1 | 2/4 | 2/4 | 4/4 | 4/4 |

All 32 arm-games completed with zero caps/errors. The standalone frozen
`GuidedSearchAgent` produced the **same winner, candidate VP, action count,
and turn count in every game** as the all-phase wrapper on this board.

Seat-level wins (each entry is one game, so these are descriptive only):

| Mix | Seat | Fair | Opening | Turn | All |
|---|---:|---:|---:|---:|---:|
| Diverse | 0 | 0 | 0 | 0 | 0 |
| Diverse | 1 | 0 | 0 | 1 | 1 |
| Diverse | 2 | 1 | 1 | 1 | 1 |
| Diverse | 3 | 0 | 0 | 1 | 1 |
| Strong | 0 | 1 | 1 | 1 | 1 |
| Strong | 1 | 0 | 0 | 1 | 1 |
| Strong | 2 | 1 | 1 | 1 | 1 |
| Strong | 3 | 0 | 0 | 1 | 1 |

The [matched smoke summary](runs/phase_guided_compare_smoke_1.json) checks
that board, seat, candidate color, rules, and opponent assignments match
across arms; it also contains paired outcome and VP differences. With one
board, its bootstrap intervals provide no useful uncertainty estimate.
Five focused tests pass for routing, forced actions, legal-action validation,
common frozen configs, and rejecting mismatched opponent assignments.

## Reproduce the 16-board screen

Run all four commands from the workspace root, keeping seed and board count
identical. These commands write the published 16-board reports; use a fresh
seed range and new report paths for further selection or confirmation.

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/mixed_eval.py --agent-file catanLabs/Competitive/phase_guided_agent.py --checkpoint catanLabs/Competitive/configs/phase_guided_opening.json --boards 16 --seed 20273800 --max-game-seconds 20 --report catanLabs/Competitive/runs/phase_guided_opening_16.json
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/mixed_eval.py --agent-file catanLabs/Competitive/phase_guided_agent.py --checkpoint catanLabs/Competitive/configs/phase_guided_turn.json --boards 16 --seed 20273800 --max-game-seconds 20 --report catanLabs/Competitive/runs/phase_guided_turn_16.json
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/mixed_eval.py --agent-file catanLabs/Competitive/phase_guided_agent.py --checkpoint catanLabs/Competitive/configs/phase_guided_all.json --boards 16 --seed 20273800 --max-game-seconds 20 --report catanLabs/Competitive/runs/phase_guided_all_16.json
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/mixed_eval.py --agent-file catanLabs/Competitive/fair_value_agent.py --checkpoint catanLabs/Competitive/configs/fair_army_finish.json --boards 16 --seed 20273800 --max-game-seconds 20 --report catanLabs/Competitive/runs/phase_guided_fair_16.json
start/.venv/bin/python catanLabs/Competitive/phase_guided_compare.py --opening catanLabs/Competitive/runs/phase_guided_opening_16.json --turn catanLabs/Competitive/runs/phase_guided_turn_16.json --all catanLabs/Competitive/runs/phase_guided_all_16.json --fair catanLabs/Competitive/runs/phase_guided_fair_16.json --output catanLabs/Competitive/runs/phase_guided_compare_16.json
```

The full guided agent scores a setup root using its best second action but
does not commit to that continuation. A separate one-board, four-seat
[follow-up probe](runs/phase_guided_followup_probe_1.json) on fresh seed
**20273900** found 100 selected expanded roots; 46 had an improving searched
follow-up. Guided's next decision matched the searched action in **44/46**
copied continuations and diverged in two: trade→settlement became another
trade, and a Road Building continuation chose a different road. All four
games completed, and all 46 copied next choices matched the actual next
actions; the probe did not advance the live RNG. Such divergence is
possible, but this small sample does **not** support it as the main cause of
the mixed-strong regression. The next intervention should target opening
quality directly while preserving FairValue as incumbent.
