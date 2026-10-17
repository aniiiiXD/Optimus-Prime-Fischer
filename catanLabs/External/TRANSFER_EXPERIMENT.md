# Porting the upstream PPO policy into Catanatron

This experiment tests whether the CPU-trained upstream PPO **policy network**
can choose legal actions in our fair, first-to-10 Catanatron arena. The adapter
is [`catanatron_transfer_agent.py`](catanatron_transfer_agent.py); it loads
[`cpu10_finetuned.pt`](models/cpu10_finetuned.pt). It does not port upstream
AlphaBot's full-game rollout search, which starts from hidden hands and deck
information.

## Adapter

- Finds an isomorphism between the engines' 19-tile, 54-vertex, 72-edge board
  graphs and selects the orientation with the most matching port sites.
- Rebuilds the upstream 1,350-feature input from the Catanatron board, public
  player scores/card **counts**, own hand, own development cards, bank, and
  turn phase. It never sends opponent resource types or development identities
  to the network. Catanatron has no count-only API, so the adapter sums those
  private arrays only to produce the publicly visible total counts.
- Translates Catanatron legal moves into the upstream 299-action vocabulary,
  scores only those moves, and returns one of the given legal objects.
  Catanatron combines robber tile and victim into one action; the network
  scores the tile and a small public-information heuristic breaks victim ties.
  A one-card Year of Plenty edge case maps to its nearest upstream pair logit.
- Domestic trade proposals are absent from the current Catanatron benchmark.
  The upstream learned policy can use those moves in its own engine, so this
  test necessarily masks them.

The [`verification script`](verify_catanatron_transfer.py) tested 227 decisions
across three seeds, covering 13 legal action types. Perturbing opponents'
private card composition, development-deck order, and RNG produced identical
observations and decisions; all selected actions were legal. It is a practical
visibility test, not a complete formal proof of game equivalence.

## Matched 10-point benchmark

Protocol v5, 16 board seeds from `20265020`, each in all four seats against
three identical opponents. Both candidates used the same seeds and limits.
All 256 games completed, with no errors. Intervals below are paired
board-bootstrap intervals for **transfer minus incumbent**.

| Opponents | Transferred PPO policy | Fair-value incumbent | Paired win-rate difference, 95% interval |
|---|---:|---:|---:|
| 3 × Catanatron Value | 8/64 (12.5%) | 21/64 (32.8%) | −20.3 points [−34.4, −6.2] |
| 3 × Catanatron AlphaBeta1 | 8/64 (12.5%) | 19/64 (29.7%) | −17.2 points [−21.9, −10.9] |

The transferred policy's median measured decision time was about 0.06 ms;
the incumbent's was about 0.3 ms. This network is fast, but substantially
weaker in Catanatron. The [transferred-policy report](reports/catanatron_transfer_port_aligned_16.json),
[incumbent report](reports/catanatron_incumbent_16.json), and
[paired comparison](reports/catanatron_transfer_port_aligned_vs_incumbent_16.json)
contain every game and the latency distributions.

An earlier, non-port-aligned orientation also scored 8/64 in both tiers, but
its report is an exploratory mapping screen. Under that orientation, the
upstream shipped weights scored 7/64 against Value and 8/64 against
AlphaBeta1; the CPU fine-tune did not show a meaningful cross-engine gain.
Those earlier reports are preserved separately.

## Interpretation

The upstream 85.4% result relied on its **own engine**, a 10-point game,
trade-enabled full-game search, and opponents with a simple trade-acceptance
rule. This Catanatron result transfers only the learned policy, because that
search would reveal hidden state. The two engines also differ in their fixed
port layouts, turn/action details, and trade interface. The board has only
three matching port sites even in the closest orientation. A direct policy
port therefore does not establish cross-engine strength.

The fair-value Catanatron agent remains the incumbent. The next useful
learning experiment is to train and validate with **the target engine's
observation and legal-action contract**, using stronger teacher trajectories
and held-out games. If upstream-style planning is added, it must sample
hidden information from a public belief state rather than clone the true
private state.

## Reproduce

From the workspace root:

```bash
start/.venv/bin/python catanLabs/External/verify_catanatron_transfer.py
start/.venv/bin/python 'catanLabs/Benchmarks&Evals/run.py' \
  --agent-file catanLabs/External/catanatron_transfer_agent.py \
  --checkpoint catanLabs/External/models/cpu10_finetuned.pt \
  --opponents value,alphabeta1 --boards 16 --seed 20265020 \
  --max-game-seconds 120 \
  --report catanLabs/External/reports/reproduced_catanatron_transfer.json
```
