# Gate 1: held-out action diagnostics

This is an imitation check on the 20-second CPU checkpoint, **not a game-strength result**. The 100-game dataset has 29,676 exact-unique decisions. The held-out set is games 80–99: 6,093 decisions from 20 complete games. The teacher is the fair Catanatron value agent under the local four-player, first-to-10, no-domestic-trade protocol.

## Where collection time went

The capped collection completed all **100 games in 55.288 seconds**. Its
[manifest](dataset.json) measured 22.629 seconds scoring fair-teacher actions,
17.632 seconds choosing behavior actions, 10.309 seconds building and encoding
visible observations, and 0.533 seconds advancing the game. These components
account for most, but not all, wall time. Game stepping alone was under 1% of
elapsed time. The phase distribution was 22,384 main-turn decisions, 4,113
discards, 1,579 robber decisions, and 800 of each opening placement type.

The 20-second CPU update used a 13,857-parameter, one-layer Transformer and
processed this dataset for two epochs in 21.253 seconds including validation
inside the measured loop. It is a profiling probe, not the 600-second model
comparison. The [four-seat game smoke](agent_smoke.json) completed four games
against Value with zero errors or illegal moves, 0/4 wins, and warm decision
latency p95 of 0.842 ms over 309 calls. Four games are too few to infer playing
strength.

## Agreement on held-out games

The model chose the teacher's exact legal action in **4,068/6,093 decisions (66.8%)**. Of these rows, **2,675 have only one legal action**. On the 3,418 decisions with a choice, exact agreement is **40.8%**.

| Phase | Decisions | Exact agreement | Agreement when choice exists |
|---|---:|---:|---:|
| Initial settlement | 160 | 18.8% | 18.8% |
| Initial road | 160 | 39.4% | 39.4% |
| Main turn | 4,602 | 79.0% | 49.9% |
| Discard | 810 | 35.2% | 34.7% |
| Move robber | 361 | 15.5% | 15.5% |

By teacher-selected action type, agreement is 49.2% for settlements (299 rows), 20.1% for roads (617), 12.3% for maritime trades (359), and 0% for buying development cards (141). These are exact-action comparisons; different legal moves can be strategically similar. Rare types such as Monopoly (13) and Year of Plenty (8) have too few examples for a reliable conclusion. The complete counts are in [diagnostics.json](diagnostics.json).

Two concrete disagreements from held-out game 80:

1. **Opening settlement, dataset row 23,583:** the teacher chose node 16; the model chose node 8. There were 54 legal settlement nodes, including nodes 0–4. This does not prove node 8 was a bad play.
2. **Main turn, dataset row 23,602:** the teacher chose road edge 5; the model chose road edge 8. There were nine legal actions, including road edges 9, 10, 11, 29 and 35. This does not prove edge 8 was a bad play.

## CPU latency and metric correction

On 200 sampled held-out decisions, after 20 warmups with four PyTorch threads, **single-example model-forward latency was 0.291 ms median and 0.336 ms p95**. This excludes visible snapshot construction, feature encoding, and action sorting. Batched evaluation of all 6,093 held-out rows took 1.30 seconds.

The original `training.json` reports 35.4% validation accuracy because its accuracy calculation takes `argmax` across padded action slots without applying the legal-action mask. Repeating that calculation gives exactly **35.3849%**; masking padded slots gives **66.7651%**. The reporting code has since been corrected and a regression test added. Training loss already masks padding, so this was a **reported accuracy bug**, not proof that the trained weights are corrupt. The 40.8% nonforced agreement is the more useful imitation diagnostic. Neither accuracy figure measures win rate.

## Reproduce

```bash
start/.venv/bin/python catanLabs/Transformer/diagnose.py \
  --dataset catanLabs/Transformer/runs/gate1_profile/dataset.npz \
  --checkpoint catanLabs/Transformer/runs/gate1_profile/short_update/best.pt \
  --training-summary catanLabs/Transformer/runs/gate1_profile/short_update/training.json \
  --output catanLabs/Transformer/runs/gate1_profile/diagnostics.json
```

Dataset SHA-256: `0716a25b8883aa32179e65009f885ddfe9c3a5dbfcdeaf4ee4bb4c15907e436d`  
Checkpoint SHA-256: `fb222107d394bbbfd589f0aeb26c9d2731a8ef2a2e53626f3e492ec29125072f`
