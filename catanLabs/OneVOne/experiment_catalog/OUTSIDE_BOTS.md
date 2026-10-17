# 1v1 reference and outside bots

These bots were opponents or outside implementations under our local
two-player Catanatron contract. They are not additional trained Hyde agents.

## Pinned Catanatron references

The [`OneVOne/evaluate.py`](../evaluate.py) harness can instantiate Random,
Weighted Random, ValueFunction, AlphaBeta depth 1, AlphaBeta depth 2,
and the Catanatron **web bot**. The web bot is a separate pruning-enabled
depth-2 AlphaBeta configuration. The harness also exposes our FairValue,
TacticalVP, and OneVOne code as optional opponents, but the first two were
developed for four-player play.

The StrategyOne dataset used **existing Catanatron** Value Base, Value
Contender, and Victory Point as label teachers. Its evaluation league also
contained Weighted Random and Random. Teacher-action accuracy means agreement
with these policies, not objective move correctness.

Local [AlphaBeta reproduction](../../FromScratch/StrategyOne/experiments/alphabeta_web/README.md)
tested 16 boards from both seats per row. Web/pruned AlphaBeta won 8/32
against Value Base and 17/32 against Value Contender; official/unpruned
AlphaBeta won 14/32 and 16/32 on the same opponent types. The unpruned
variant won 12/32 against OneVOne. Pruning changed play; 32 games per row
cannot establish a stable global ranking. Its internal simulator search
may access hidden state.

## Published HexMachina player and mechanism port

Source: [local reproduction report](../../FromScratch/StrategyOne/experiments/hexmachina_web/README.md).
We tested the author's released `FooPlayer` through a compatibility adapter
and a separate local approximation of its described phase/rollout mechanism.
On 16 boards × both seats, the published player won **17/32** against the
web bot and **14/32** against OneVOne. The mechanism port won **16/32**
against the web bot. Those small samples are comparison probes, not proof
that either agent outperforms our incumbent.

The published code's direct player-field lookups did not match the provided
engine's state shape, so its phase and resource heuristic were inactive in
our local run after placement. The approximation does not reproduce the
paper's evolution process. Both full-state simulator paths have hidden-state
fairness limits. Keep the two implementations distinct.

## External projects not run as local 1v1 bots

BenjaminL1's Colonist-style 1v1 architecture is a design reference, not
a measured opponent here. Eli6th's headline PPO/AlphaBot results came from
its own four-player engine and are outside this 1v1 catalog. See
[`catanLabs/LATER.md`](../../LATER.md) for that comparison.
