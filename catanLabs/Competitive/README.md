# Competitive CPU Catan agent

The current promoted four-player policy is
[`tactical_vp_agent.py`](tactical_vp_agent.py) with
[`tactical_vp_min0.json`](configs/tactical_vp_min0.json). Its confirmed
results are in [`TACTICAL_VP_MIN0_CONFIRM.md`](TACTICAL_VP_MIN0_CONFIRM.md).
The earlier [`fair_value_agent.py`](fair_value_agent.py) with
[`fair_army_finish.json`](configs/fair_army_finish.json) remains the matched
control. That policy uses a
one-action value search, expected value over all 11 possible dice sums, an
unknown-card estimate for development purchases, public production estimates
for Monopoly, and a robber score covering all three opponents. It never asks
the simulator for the next roll, the hidden development-card order, or an
opponent's resource composition.

The first Transformer remains a separate learning baseline. This policy is a
stronger teacher and CPU yardstick for later model work. No neural weights are
required at inference time.

## New policy-guided search test

[`SEARCH_EXPERIMENT.md`](SEARCH_EXPERIMENT.md) reports a two-action search
layer over this policy, plus a version using the pilot Transformer to nominate
search branches. Across 48 new boards and four seats, the fair search won
**88/192 against three AlphaBeta1 bots**, versus **64/192** for this policy.
Its mixed-strong result was 36/96 versus 38/96, so it remains an experimental
specialist rather than replacing the general incumbent. The Transformer prior
showed no clear strength gain over fair-policy search.

Run the search version from the workspace root:

```bash
start/.venv/bin/python 'catanLabs/Benchmarks&Evals/run.py' \
  --agent-file catanLabs/Competitive/guided_search_agent.py \
  --checkpoint catanLabs/Competitive/configs/guided_search_3.json \
  --opponents value,alphabeta1 --boards 24 --seed 20262300 \
  --report catanLabs/Competitive/runs/reproduced_search.json
```

## Run it

From the workspace root:

```bash
start/.venv/bin/python 'catanLabs/Benchmarks&Evals/run.py' \
  --agent-file catanLabs/Competitive/fair_value_agent.py \
  --checkpoint catanLabs/Competitive/configs/fair_army_finish.json \
  --opponents random,weighted,value,alphabeta1 --boards 24 \
  --seed 20261900 \
  --report catanLabs/Competitive/runs/new_eval.json
```

The benchmark protocol is four players to 10 points, no player-to-player
offers, seat rotation, and a legal-action check. Protocol v5 isolates RNG in
hypothetical copies, fixes forced chance outcomes in upstream search, and
locks Python's hash seed plus legal-action order for repeatable ties.
`create_agent(color,
checkpoint, seed)` loads the JSON parameters and returns an object with
`decide(game, playable_actions)`.

To check mixed opponent tables with the same chance and ordering fixes:

```bash
start/.venv/bin/python catanLabs/Competitive/mixed_eval.py \
  --agent-file catanLabs/Competitive/fair_value_agent.py \
  --checkpoint catanLabs/Competitive/configs/fair_army_finish.json \
  --boards 24 --seed 20261950 \
  --report catanLabs/Competitive/runs/new_mixed_eval.json
```

## What changed from Catanatron Value

| Decision | Policy behavior |
|---|---|
| Initial and deterministic moves | Score every legal action using a copied state and a value function over own points, production, reachability, hand, roads, and army. |
| Roll | Average over the 11 dice sums with their correct probabilities. |
| Buy development card | Value an unknown card by its count and resources spent, without looking at the deck order. |
| Monopoly | Estimate other players' holdings from public production. |
| Move robber | Prefer blocking high-value enemy production and threatening leaders while avoiding our own production. |

The selected parameters increase the value of reachable settlement sites and
hand combinations. They also value played Knights and give buying a
development card a modest bonus once our actual score reaches seven points.
The latter two choices improved the race to 10 points against the stronger
local bots. The configuration is deliberately kept as JSON for reproducible
parameter searches.

## Verification and evidence

- [`RESULTS.md`](RESULTS.md) gives the seeded tournament table, paired
  comparison, decisions, and limits.
- [`check_visibility.py`](check_visibility.py) perturbs hidden opponent hands,
  development deck order, and RNG while preserving the visible position. The
  selected agent made the same choice in over 100 checked decisions, including
  a synthetic state with Roll, Knight, and Monopoly available.
- [`compare.py`](compare.py) computes board-clustered bootstrap intervals for
  paired seat and seed results.
- The complete JSON reports are in [`runs/`](runs/).

These tests are local Catanatron results. The benchmark still passes a full
game object, so the visibility check supports the claim of fair behavior but
is not a formal observation-only interface. Domestic trade offers are outside
the current action protocol.
