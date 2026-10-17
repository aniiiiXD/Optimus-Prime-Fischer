# OneVOne: how it wins and where it loses

Measured on 2026-10-17. This diagnoses the frozen official-game baseline; it does
not introduce or train OneVOne V2.

## Test and result

**127 wins, 73 losses, no draws: 63.5% against the official F agent.**
The 200 games use 100 boards, seeds 2026101800–2026101899, each played in both
seats. This is a replay of the baseline evaluation, not 200 new independent
games or a test against every competitor. First-seat wins: 65/100; second-seat
wins: 62/100.

An engine-side observer reproduced the original win count and all 9,934 agent
decisions. Agent files still match the recorded baseline SHA-256 hashes.
Official clocks remained enabled: zero errors, crashes, timeouts, random
fallbacks or budget overruns for either side. This rules out those technical
failures in this run; organizer-machine timing remains unmeasured.

Sources: [baseline manifest](reports/onevone_baseline/baseline_200.json),
[trace summary](reports/onevone_analysis/baseline_trace_200/summary.json),
[per-game records](reports/onevone_analysis/baseline_trace_200/games.json),
[derived metrics](reports/onevone_analysis/baseline_trace_200/metrics.json).

## What differs between wins and losses

These are averages grouped by eventual outcome. They locate problems but do
not isolate whether a decision, board, dice, or opponent response caused them.
“Turn” is the engine's counter across both players, not our own-turn count.

| Measurement | Our 127 wins | Our 73 losses |
|---|---:|---:|
| First additional settlement, mean turn when built | 25.9 | 32.8 |
| Games with no additional settlement | 0/127 | 7/73 |
| First city, mean turn when built | 26.4 | 32.8 |
| Games with no city | 6/127 | 10/73 |
| Our / F's VP at turn 20 | 2.99 / 2.43 | 2.47 / 2.97 |
| Our / F's expected cards per roll at turn 20 | 0.871 / 0.694 | 0.713 / 0.855 |
| Our / F's resources received from rolls per game | 70.9 / 52.2 | 54.1 / 72.5 |
| Our development-card purchases per game | 2.66 | 0.53 |
| Games ending with our Longest Road | 92/127 | 5/73 |
| Games ending with F's Longest Road | 28/127 | 68/73 |
| Our final VP, mean | 10.14 | 5.97 |

First-settlement timing excludes the seven losses where none was built. City
timing includes 121 wins and 63 losses where a city was built. All 200 games
reached the turn-20 checkpoint. Expected production uses dice probabilities
and buildings, before robber blocking and bank shortages.

### How we win

1. **An early building lead becomes a production lead.** In wins, we add a
   settlement around turn 26 versus F's turn 33 and build the first city around
   turn 26 versus F's turn 33. More production makes further builds affordable.
2. **We have several ways to finish.** The final actions in our 127 wins were
   43 settlements, 21 cities, 14 roads, 25 development purchases that drew a VP
   card, and 24 knight plays that claimed Largest Army. Thus 49/127 wins ended
   through development cards. These are final actions, not the entire VP mix.
3. **Longest Road often contributes two points.** We held it in 92/127 wins;
   holding it at the end is an association, not proof that more road spending
   would always improve the agent.
4. **The tactical layer helps on this screen.** Removing it lost nine net wins
   over the same 200 games; the controlled comparisons below give uncertainty.

### Where we lose

1. **Most losses develop before the final point.** In 58/73 losses we finished
   at seven VP or fewer; only 15 ended at eight or nine. A turn-20 production
   deficit is already visible. Fixing last-move mistakes alone cannot explain
   or recover all these losses.
2. **Expansion and city development lag.** In losses our first extra settlement
   arrives around turn 33 versus F's turn 26. Our first city arrives around
   turn 33 versus F's turn 27. We collect fewer resources and buy fewer cards.
   Resource balance, placement, dice, and spending decisions all remain possible
   contributors; this audit does not assign a percentage of blame to each.
3. **F usually controls Longest Road when it wins.** It holds the award in
   68/73 losses. This makes route competition worth examining; blindly chasing
   roads can itself delay productive buildings.
4. **The scorer has limited planning depth.** It normally scores one successor
   state. Its tactical override examines two of our actions, with restricted
   follow-up types, and does not search an opponent reply. It therefore has
   no general proof that a locally attractive move prevents an opponent win.
   This is a code-level limitation, not a measured count of resulting losses.

Two simple explanations were not supported: there were no clock failures, and
neither side ended a turn with an immediately legal city or settlement build.
Our average discards were lower in losses (6.1) than in wins (7.3), so excessive
discarding is not established as the main cause either.

## Which existing components contribute?

Four isolated copies each played the same 200-game schedule against F. The
baseline was not edited. Differences are percentage points, relative to 63.5%.

| Diagnostic copy | Wins | Win rate | Difference | Paired 95% interval for difference |
|---|---:|---:|---:|---:|
| Frozen baseline | 127/200 | 63.5% | — | — |
| Tactical override disabled | 118/200 | 59.0% | −4.5 pp | −9.5 to 0.0 pp |
| Five custom value weights replaced with defaults | 122/200 | 61.0% | −2.5 pp | −9.5 to +4.0 pp |
| Generic robber scorer substituted | 127/200 | 63.5% | 0.0 pp | −4.0 to +4.0 pp |
| Late development-buy bonus removed | 122/200 | 61.0% | −2.5 pp | −6.0 to +1.0 pp |

The intervals resample whole board pairs, 10,000 bootstrap samples. All include
or touch zero. Treat the differences as exploratory, especially with four
comparisons on one opponent and one board set. They are not additive. The
generic robber changed individual outcomes (nine wins gained and nine lost),
despite the same total. Zero net change does not establish equivalence.

All 800 component-test games completed without faults or budget overruns on
either side. [Full comparison evidence](reports/onevone_analysis/component_removals/summary.json).

## Concrete decision mistakes found

We saved 1,199 masked positions and searched 294 selected late-game positions:
ending a turn at eight or more VP, or buying a development card at nine VP.
The offline search followed at most three deterministic actions, with a
1,500-expansion cap. No search hit that cap. It did not inspect future dice,
the actual hidden card deck, or opponent hidden development cards.

### 1. A possible VP-card draw suppresses a guaranteed-win search

In [tactical_vp.py](onevone_baseline/tactical_vp.py), buying a development card
is assigned its maximum possible immediate gain of one VP. At nine VP, this
causes the tactical scan to return before checking guaranteed alternatives.

There were 99 nine-VP purchases. In ten of those positions, a deterministic
trade/build sequence could already win. Seven purchases drew knights; three drew VP cards and won immediately. All ten positions
were in games we eventually won. Example: game 32, turn 52, trade four wheat
for a brick and build settlement 27 to win; the agent instead bought a knight.

### 2. Two-action search cannot complete a three-action Road Building win

Three END_TURN decisions, across games 14 and 72, had a guaranteed line:
play Road Building, place road one, place road two, claim Longest Road and win.
The existing two-action search cannot evaluate the entire sequence. We
eventually won both games, so this audit measures delayed wins, not lost games.

Together these are **13 positions across 11 games** with a guaranteed winning
alternative. None was in the 73 losses. The bounded search finding no such
line in a loss does not prove that deeper or differently selected lines do
not exist.

### 3. Monopoly estimates a hand that the official game makes public

In [fair_value.py](onevone_baseline/fair_value.py), Monopoly uses a capped
production-based estimate of the opponent's resources instead of their exact
visible counts. Of 33 Monopoly plays, five took zero cards; nine selected a
resource scoring strictly below another legal choice when exact transfers were
evaluated with the same state scorer. All these cases occurred in winning games.

Example: game 7, turn 34, the opponent had four wood and zero ore. The policy
selected ore, estimating two cards, and actually took zero. Four wood was a
better legal alternative under its own scorer. A better state score is not
itself proof of a higher eventual win probability.

[All position checks and sequences](reports/onevone_analysis/baseline_trace_200/position_checks.json).

## Code-level explanation to test next

The inherited value function gives public VP a weight of `3e14`, production
`1e8`, and much smaller weights to resource combinations and reachable sites.
It sums resource production with a variety bonus, while hand synergy averages
distance to a city and a settlement. This is a coarse economy model: it does
not calculate expected turns to a chosen build sequence. Its very strong
immediate-VP preference can also favor a point now over a better investment.
Both behaviors are visible in the code; neither has yet been isolated as the
cause of the early deficits.

For V2, the evidence supports this order of experiments:

1. Fix exact-information and guaranteed-win handling independently: Monopoly
   transfers, certain wins before speculative card purchases, and complete
   Road Building sequences. These have specific reproducible examples, but no
   measured win-rate gain yet.
2. Investigate the larger early-game deficit: target-specific resource
   bottlenecks, cost and time to the next settlement/city, and road routes that
   enable useful buildings. Compare choices from actual losing positions.
3. Test opponent immediate-win checks using visible resources, buildings and
   roads. Do not assume knowledge of hidden VP or development-card identities.
4. Keep each change separate; screen on paired boards, then confirm candidates
   on fresh boards with the official clock. No dataset or ML training is needed
   for these deterministic experiments.

## Reproduction and data limits

- [Trace recorder](analyze_onevone.py) observes the official engine outside the
  agent process. [Component runner](ablate_onevone.py) evaluates separate copies.
- [Position inspector](inspect_onevone_positions.py) searches saved masked
  states offline. Its compute is not evidence that the same search fits the
  live agent's budget.
- Rebuild aggregate metrics without playing games:
  `python3 HackathonDay/summarize_onevone_analysis.py`.
- Detailed traces include hidden identities for post-game analysis. Agents did
  not receive these fields. They are not ready-to-use policy-training inputs.
- Only official F was tested. Same-board replay and component removals do not
  establish performance against other teams, O, or fresh tournament boards.

**Current conclusion:** the agent often converts an early economic lead into
road and development-card points. Its losses mostly show an early economic
and expansion deficit. Specific tactical defects exist, but the observed
defects above do not explain most losses. V2 improvement is still unmeasured.
