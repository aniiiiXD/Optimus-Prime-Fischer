# Public history, backward counterfactuals, and bounded search

Branch: `experiment/catan-longer-training-heuristics`  
Worktree: `/Users/anixd/Documents/Hyde-catan-experiments`

All changes and outputs stayed in the isolated worktree. The original agent
and checkpoints in `/Users/anixd/Documents/Hyde` were not changed.

## Experimental contract

Two-player Catanatron, 10 VP, the frozen calibrated MLP plus four-epoch
settlement head as the current agent. The same seed, opponent, and seat were
used for each paired comparison. `install_v5_protocol()` isolates copied-game
RNG and canonicalizes legal-action order. The fresh evaluation seeds below do
not overlap the existing imitation dataset or training games.

### Public-history Transformer

The Transformer reads 16 recent public actions, actor identity, and public
dice results. A state-only residual is the control. Both train for three
epochs on the same 12,000 sampled teacher decisions; a separate 3,000
decision game-disjoint sample is held out. The original calibrated policy is
frozen. Teacher matching is a diagnostic; winning is the adoption test.

| Arm | Held-out teacher match | Fresh wins, seed 20267300 | p95 decision time |
| --- | ---: | ---: | ---: |
| Current agent | 52.3% | **36/64** | 0.72 ms |
| State-only residual | 55.4% | 20/64 | 1.01 ms |
| History Transformer, full residual | 55.6% | 26/64 | 1.41 ms |
| History Transformer, 0.25 residual | — | 29/64 | 2.22 ms |
| History Transformer, 0.50 residual | — | 25/64 | 2.22 ms |

The Transformer beat its retrained control by six games, but lost ten games
relative to the current agent. Lowering its influence did not recover the
baseline. This tests a small, teacher-trained, 16-event model; it does not
rule out a history model trained on win outcomes or much longer context.

### Bounded same-turn search

When our actual VP is at least six, breadth-first search checks deterministic
builds and maritime trades for a sequence that wins before ending the turn.
Depth is four and at most 80 copied states are searched per decision. The
opponent's hidden hand and future dice are not used. This is a tactical search,
not general MCTS. It remains an optional `--same-turn-search` switch.

| Fresh seed range | Games | Current wins | Search wins | Paired gains | Paired losses | Search changed decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 20267300–20267363 | 64 | 36 | 38 | 2 | 0 | 56 |
| 20267500–20267563 | 64 | 37 | 38 | 1 | 0 | 44 |
| 20267600–20267727 | 128 | 63 | 64 | 1 | 0 | 73 |
| 20267800–20268055 | 256 | 133 | 136 | 3 | 0 | 166 |
| **Total** | **512** | **269 (52.5%)** | **276 (53.9%)** | **7** | **0** | **339** |

By opponent: Value Base 10→12 wins in 104 games; Value Contender 12→15 in
103; Victory Point 84→84 in 103; Weighted Random 79→81 in 102; Random
84→84 in 100. The search gain is small but consistently nonnegative on these
four screens. Seven paired flips all favored search; these experiments were
part of a wider search over ideas, so that pattern is not a final statistical
guarantee. Across screens, p95 decision time ranged from 2.0 to 6.4 ms with
search versus 0.5 to 1.3 ms without it; machines ran other experiments at
different times, so treat these as observed ranges rather than a controlled
latency benchmark.

### Backward loss audit and root UCB

`loss_counterfactual_search.py` played six games against Value Base and worked
backward from losses. It examined 13 earlier decision states. At each state it
compared the played move with up to two alternatives using 12 root-UCB
simulations, a 150-ply horizon, and eight separate paired chance rollouts for
validation. If no terminal win occurs by the horizon, the leaf score uses VP
and production gap.

The paired validation favored an alternative in three of 13 states. Two were
earlier End Turn versus maritime trade choices. One late state favored a
maritime trade slightly over a road. Most alternatives still led to losses;
one changed move did not fix the downstream policy. The first audit at the
very last decisions found zero promising alternatives in four losses, which
is why the later audit moved to earlier milestones and settlement choices.

**Oracle limitation:** these offline game copies contain the opponent's true
hidden resources and development cards. Their branch values cannot be used as
live-policy performance or as leakage-free training labels. This is a
retrospective diagnostic and a prototype of root-level UCB, not full-tree
MCTS. A live search would need to sample plausible hidden states from public
history, then test win rate under the same sealed evaluation contract.

## Verdict

Keep the small deterministic same-turn search as a promising optional
experiment. It gained seven wins in 512 matched games, including five against
the two strongest Value bots, at extra decision cost. The present Transformer
and the offline oracle search should not replace the current agent. The next
substantial improvement needs a stronger continuation policy or a legal
belief-state search over opponent threats; this audit did not establish either.
