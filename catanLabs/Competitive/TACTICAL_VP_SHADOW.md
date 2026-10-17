# G5 late-turn VP search: two-board shadow probe

This is a research candidate, not a promoted agent. [`tactical_vp_agent.py`](tactical_vp_agent.py) inherits the selected FairValue policy and keeps its choice unless an exact deterministic action, or a two-action sequence, scores more immediate VP. It searches only after the actor has rolled and has at least 7 actual VP. Roots are bank trades, roads, settlements, cities, Year of Plenty, and Road Building; second actions are roads, settlements, and cities. It copies game states without executing rolls, theft, Monopoly, or unknown development-card purchases. A FairValue Knight play is safely simulated only up to the public Largest Army award, before the robber prompt; an unknown development-card purchase gets a conservative one-VP upper bound. All copied legal actions execute with rule validation. The fixed budget is 64 roots and 256 followups; an immediate win ends the search.

The old [`guided_search_agent.py`](guided_search_agent.py) expands only the top three roots of each action type by immediate FairValue score. In the v2 100-game dataset, 1,422 late `PLAY_TURN` rows at public VP ≥7 had bank trades available; all had more than three, with 5.97 on average and at most 20. A lower-ranked trade can unlock a city, settlement, or Longest Road point. The previous broad two-action search beat FairValue against uniform AlphaBeta1 but did not beat it against the strong mixed table ([earlier experiment](SEARCH_EXPERIMENT.md)). This candidate tests a narrower tactical mechanism.

## Shadow result

The [JSON report](runs/tactical_vp_shadow_2boards.json) covers seeds **20271200–20271201**, all four candidate seats, against the strong mix (Value, AlphaBeta1, AlphaBeta1). **FairValue played every actual candidate move**; the scanner only reported alternatives. All eight games completed. Across 784 FairValue decisions, 84 were eligible for this scanner. It found an immediate VP line in 13 and would have changed FairValue's first move in 6. No scan hit its fixed budget. The eight sampled hidden-hand/deck perturbations preserved FairValue and scanner choices, and no shadow decision advanced the live game's RNG. Five focused tests pass, including a winning Knight guard, hidden purchase upper bound, and validated hypothetical actions.

The clearest example is seed **20271200, seat 1, turn 87**: at 8 VP FairValue chose `END_TURN`. Trading two wheat for brick, then building road `(52, 53)`, yielded 10 VP through Longest Road in the copied state. FairValue did later win this same game, but the shadow line would have finished it four turns earlier if played. Other proposed lines were bank trade → settlement at 7–8 VP. These are legal hypothetical continuations, not measured win-rate improvements; the actual games never followed the scanner.

Scanner-only latency on the 84 eligible decisions was **2.06 ms median, 25.85 ms p95, 59.78 ms max**. The long tail came from states with many trade and road combinations. This is the main CPU risk; there may also be a strategic cost when a quick VP sacrifices stronger production. A larger tournament should wait until the DAgger run finishes. The next gate is a matched 24-board, all-seat screen against FairValue in both mixed compositions, recording win rate, mean VP, illegal actions, and decision latency; advance only if the strong mix improves without unacceptable tail latency, then confirm on fresh boards.

Reproduce the bounded probe:

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/tactical_vp_shadow_probe.py \
  --boards 2 --seed 20271200 --max-game-seconds 20 \
  --report catanLabs/Competitive/runs/tactical_vp_shadow_2boards.json
```
