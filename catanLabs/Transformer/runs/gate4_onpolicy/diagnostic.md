# On-policy Transformer diagnostic

The trained Transformer played these native Catanatron games. At each of its visited decisions, the fair-value teacher independently chose from the **same legal actions in the same state**. Disagreement is a diagnostic, not proof that either move is better.

Protocol: `catanLabs-four-player-v5-canonical-actions-and-chance`; opponents: value, alphabeta1, fair; games: **4**; complete: **4**.

| Phase | Candidate decisions | Exact teacher match | Match when choice exists |
|---|---:|---:|---:|
| BUILD_INITIAL_ROAD | 8 | 50.0% | 50.0% |
| BUILD_INITIAL_SETTLEMENT | 8 | 0.0% | 0.0% |
| DISCARD | 37 | 40.5% | 40.5% |
| MOVE_ROBBER | 11 | 18.2% | 18.2% |
| PLAY_TURN | 172 | 84.3% | 60.3% |

## Opening and VP traces

- Seed 20269200, seat 0: completed; candidate final VP 4; opening production 0.528 cards/roll across 2 resource types. VP trajectory: [(0, 0), (7, 2), (14, 3), (21, 4), (28, 4), (35, 4), (42, 4), (49, 4), (56, 4), (63, 4), (70, 4)].
- Seed 20269201, seat 1: completed; candidate final VP 2; opening production 0.472 cards/roll across 3 resource types. VP trajectory: [(0, 0), (8, 2), (16, 2), (24, 2), (32, 2), (40, 2), (48, 2), (56, 2), (64, 2), (72, 2), (80, 2)].
- Seed 20269202, seat 2: completed; candidate final VP 5; opening production 0.472 cards/roll across 4 resource types. VP trajectory: [(0, 0), (6, 2), (12, 2), (18, 3), (24, 3), (30, 5), (36, 6), (42, 6), (48, 6), (54, 6), (60, 7)].
- Seed 20269203, seat 3: completed; candidate final VP 2; opening production 0.528 cards/roll across 3 resource types. VP trajectory: [(0, 0), (7, 2), (14, 2), (21, 2), (28, 2), (35, 2), (42, 2), (49, 2), (56, 2), (63, 2), (70, 2)].

The JSON report contains each candidate decision, its legal count, model and teacher choices, own resource hand, opening-node production proxies, and the turn-by-turn VP trajectory. The production proxy omits road access, contention and resource scarcity. A few games cannot establish playing strength or cause of loss.

## Reproduce

```bash
start/.venv/bin/python catanLabs/Transformer/on_policy_diagnostic.py --games 4 --seed 20269200 --checkpoint /Users/anixd/Documents/Hyde/catanLabs/Transformer/runs/gate3_matched/transformer/best.pt --max-total-seconds 50.0 --max-game-seconds 22.0 --max-turns 500 --output catanLabs/Transformer/runs/gate4_onpolicy/diagnostic.json
```

Checkpoint SHA-256: `7a28240e136bbcf62e597a4131c4598241e451a803889b8d28e396e77cbf6081`  
Rules-manifest SHA-256: `6996bcc8f1680bbb483f64cf7281305b925bc97974e50ef8136d25d66ce2bf3e`
