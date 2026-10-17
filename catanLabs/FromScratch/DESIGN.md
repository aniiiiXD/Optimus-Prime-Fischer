# Design: Catan From Scratch

## Objective

Build an agent that wins 1v1 Catan matches under the target competition's rules
and action interface. The announcement says it will face organizer reference
bots and agents from other teams. It also requires us to explain the strategy
clearly. Training loss and imitation accuracy are diagnostics; match results
decide whether a change is useful.

## Competition constraints from the announcement

The organizer calls out dice randomness, hidden information, uncertain trades,
and limited compute. It permits search, self-play, learned value functions,
imitation learning, handcrafted heuristics, or other approaches. The judging
criteria are consistent wins and a clear explanation of the strategy. The full
problem statement, compute constraints, submission rules, and evaluation
details will be provided at the event, so the agent interface and final
benchmark protocol remain open.

## Initial rules baseline from the Hyde deck

The [Hyde@IITB introductory deck](../../../../Downloads/Hyde@IITB.pdf) says two
players compete on a resource island and the first to 10 victory points wins.
It describes five resources: wood, brick, sheep, wheat, and ore. Numbered hexes
produce resources when their number is rolled; settlements collect one and
cities collect two from adjacent producing hexes. These are the rules the deck
states, recorded here as a provisional baseline.

| Action or scoring item | Deck description |
| --- | --- |
| Road | 1 wood + 1 brick; connects the player's network; 0 VP |
| Settlement | 1 wood + 1 brick + 1 sheep + 1 wheat; collects resources; +1 VP |
| City | 3 ore + 2 wheat; upgrades a settlement; +2 VP |
| Development card | 1 ore + 1 sheep + 1 wheat |
| Knight | Move the robber and potentially steal a resource; 3 knights qualify for Largest Army (+2 VP) |
| Road Building | Place 2 free roads |
| Year of Plenty | Take any 2 resources |
| Victory Point card | Hidden card worth +1 VP |
| Robber | A 7 activates it; it blocks production on its tile and may steal from an opponent |
| Trading | Bank 4:1, generic port 3:1, matching-resource port 2:1 |
| Longest Road | 5 connected segments qualify for +2 VP |

The deck does not specify starting placement, development-card deck counts or
timing, all robber/discard edge cases, trade-offer protocol, action encoding,
compute limits, or competition-specific 1v1 adjustments. Those need to come
from the official rules and interface before we treat this baseline as the
simulator contract.

## Evaluation standard

- Use fixed, recorded board seeds and play each seed from both seats.
- Keep evaluation seeds separate from training and tuning seeds.
- Test against a small frozen league, including the current baseline.
- Report wins, win rate with uncertainty, VP margin, illegal actions, failed or
  capped games, and decision latency.
- Keep an explanation of the strategy and evidence for each major design
  choice, since the competition also judges how clearly we explain the agent.
- Promote a policy only when a fresh evaluation supports the improvement.

## Build sequence

1. **Specify the game contract.** Record rules, observations, legal-action
   format, end conditions, and compute limits. Mark any provisional assumptions.
2. **Validate the simulator adapter.** Convert observations and actions, verify
   legality, and test that a seed can be replayed from both seats.
3. **Establish a baseline.** Make a simple agent that completes games reliably
   and record its paired-seat performance.
4. **Generate training data.** Save visible states, legal actions, selected
   actions, rewards/outcomes, teacher identity, seed, seat, and game status.
   Keep all positions from a board seed in the same data split.
5. **Train and compare policies.** Add imitation, value learning, search, or
   reinforcement learning as experiments with explicit baselines and held-out
   match tests.

## Data principles

- Store the information the acting player can actually observe; do not include
  hidden opponent resources or future randomness.
- Keep legal actions attached to each state so the agent can be constrained to
  moves allowed by the simulator.
- Preserve how each label was produced. A teacher's move is evidence of that
  teacher's behavior, not proof that it is optimal.
- Record terminal outcomes and game-level results separately from per-action
  targets so credit assignment stays explicit.
- Exclude incomplete games from terminal-outcome training, while retaining
  their failure reason in the manifest.

## Current status

The clean workspace and first rules baseline are documented. The official
competition API and complete rule contract still need confirmation; until then,
engine settings and omitted edge cases remain provisional.
