# 1v1 heuristics and planning: results and failure modes

## A. OneVOne heuristic/search agent

Code: [`OneVOne/agent.py`](../agent.py) with
[`configs/default.json`](../configs/default.json).
Evidence: [`TRAINING_RESULTS.md`](../TRAINING_RESULTS.md) and the isolated
worktree's `reports/INVEST_CONVERT_RESULT.md`.

It scores VP, dice-weighted production, reachable sites, hand readiness,
army/road potential, and a two-player robber target. The inherited bounded
scan checks deterministic same-turn trades/builds for VP before choosing an
action. This is a heuristic/search policy, not a neural network.

In its original paired-seat Catanatron web-bot test it won **85/128**. On
fresh planner comparison schedules it won **93/128** and **96/128** against
the same bot. It is the strongest tested 1v1 agent on those schedules.
Its mechanism gives it immediate conversion and target selection that the
MLP lacks; the result does not isolate which ingredient causes how many
wins. It has no measured event-rule or human win rate.

### Parameter evolution

Two seeded searches mutated heuristic/resource/robber/search parameters.
Neither found a held-out improvement. The second's 32-board, four-opponent,
both-seat validation was **150/256 (58.6%)** for the unchanged baseline.
This rejects those sampled parameter changes; it does not prove the default
weights globally optimal. No `configs/trained.json` was promoted.

## B. Small exact same-turn search on StrategyOne

Source in the isolated worktree:
`catanLabs/FromScratch/StrategyOne/reports/HISTORY_AND_SEARCH_EXPERIMENTS.md`.
When already at least six VP, it searches up to four deterministic trade
and build steps for an immediate win, with an 80-state budget. Across four
fresh, matched mixed-bot screens, baseline won **269/512** and the search
variant **276/512**. Seven outcomes flipped toward search, none the other
way; 339 decisions changed. The gain is real on those screens but small,
and this was one idea among several explored. Search deliberately misses
longer setups and moves requiring future dice or unknown draws.

## C. Backward loss audit and root UCB

Source in the same isolated worktree report. After six losses to Value Base,
the offline probe inspected 13 earlier decision states and found three
alternatives favored by paired rollouts. Two were End Turn versus maritime
trade choices. Most alternate lines still lost. Its copied states contained
the opponent's true hidden cards; therefore these estimates are diagnostic
oracles, **not** valid live-policy win rates or fair training labels. No
deployable MCTS agent was established by this experiment.

## D. Invest/convert planner, dynamic and fixed horizon

Code in isolated worktree:
`catanLabs/FromScratch/StrategyOne/invest_convert.py` with two JSON
profiles in adjacent `configs/`. Full evidence:
`reports/INVEST_CONVERT_RESULT.md` and `invest_convert_confirm.json`.

The calibrated StrategyOne MLP and settlement head stay frozen. The planner
searches deterministic same-turn **bank trade → build** sequences and takes
a found immediate win. It values VP, production, reach, hand, and likely
discard. The dynamic version shrinks an investment horizon as public VP
rises; the fixed version uses a constant horizon. Coefficients were selected
on a separate 16-board outcome screen.

| Fresh web-bot confirmation: same 64 boards × both seats | Wins / 128 |
| --- | ---: |
| Frozen calibrated MLP + settlement head | 12 |
| MLP + dynamic horizon | 64 |
| MLP + fixed horizon | **67** |
| OneVOne control | **93** |

**What worked:** making legal trade/build continuations explicit raised the
MLP's wins from 12 to roughly half the games. In this screen, the frozen MLP
made 22 maritime trades and 4,985 individual discard actions; the dynamic
planner made 1,105 trades and 964 discards. It also built more settlements
and cities. These counts show a large behavior change, not a full causal
breakdown of every added win.

**What failed:** the dynamic horizon scored three fewer wins than fixed
horizon on matched games. Thus the experiment supports planning and
conversion, **not** the proposed explore/exploit switching rule. The
planner still trails OneVOne substantially.

## E. OneVOne/MLP phase hybrid

Code: `factory_from_baseline(..., "hybrid")` in the isolated worktree's
`invest_convert_experiment.py`. The hybrid uses OneVOne except during
post-roll `PLAY_TURN`, where it delegates to the MLP planner. On a *second*
fresh 128-game web-bot schedule, it won **85**, versus **96** for OneVOne
and **73** for the fixed-horizon MLP planner. The hybrid overturned some
OneVOne losses but lost more of OneVOne's wins. Replacing its post-roll
choices with this planner is not a measured upgrade.
