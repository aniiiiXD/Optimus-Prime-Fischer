# Frozen Transformer phase-swap screen

This is an **eight-board exploratory ablation**, not a promotion tournament. The frozen v2 Transformer checkpoint is unchanged. Four candidates played the same Catanatron v5 first-to-10 games, each board in all four seats, against the same opponent assignments. Domestic trade offers were absent from this benchmark.

The two hybrid agents use [`opening_rescue_agent.py`](opening_rescue_agent.py):

- **Model turn only:** Transformer chooses `PLAY_TURN`; fair-value chooses opening, discard, robber and other phases. [Config](configs/phase_swap_model_turn.json).
- **Model opening only:** Transformer chooses initial settlement and road; fair-value chooses every other phase. [Config](configs/phase_swap_model_opening.json).

The controls are the full Transformer and full fair-value agent. The two opponent mixes are `mixed_diverse` (Weighted, Value, AlphaBeta1) and `mixed_strong` (Value, AlphaBeta1, AlphaBeta1). Seeds **20269300–20269307** were fresh relative to the v2 dataset and earlier on-policy diagnostic. All 256 games completed, with zero agent errors. Each game had a 20-second wall cap; the four tournament runs took about 90 seconds total. Opponent assignment, seed and seat matched exactly across candidates.

| Candidate | Diverse wins / 32 | Mean VP | Strong wins / 32 | Mean VP | Combined wins / 64 |
|---|---:|---:|---:|---:|---:|
| Full fair-value | 11 | 7.22 | 12 | 7.59 | 23 |
| Transformer turn only | 6 | 6.00 | 8 | 6.00 | 14 |
| Transformer opening only | 5 | 6.06 | 2 | 5.25 | 7 |
| Full Transformer | 1 | 4.34 | 0 | 3.44 | 1 |

Paired win-rate differences below use the same games and a **board-clustered bootstrap** over eight board seeds (95% intervals, percentage points):

| Comparison | Diverse | Strong |
|---|---:|---:|
| Turn only − fair | −15.6 [−28.1, 0.0] | −12.5 [−28.1, +3.1] |
| Opening only − fair | −18.8 [−37.5, 0.0] | −31.2 [−50.0, −9.4] |
| Full Transformer − fair | −31.2 [−46.9, −12.5] | −37.5 [−53.1, −21.9] |
| Turn only − full Transformer | +15.6 [0.0, +34.4] | +25.0 [+9.4, +40.6] |
| Opening only − full Transformer | +12.5 [−3.1, +28.1] | +6.2 [0.0, +15.6] |

The same-seed mean VP difference versus full fair-value was **−1.22/−1.59** for turn-only, **−1.16/−2.34** for opening-only, and **−2.88/−4.16** for full Transformer (diverse/strong mixes). Detailed [game reports](runs/phase_swap_model_turn_8.json), [opening-only](runs/phase_swap_model_opening_8.json), [fair control](runs/phase_swap_fair_control_8.json), [full Transformer](runs/phase_swap_full_transformer_8.json), and [paired comparisons](runs/turn_vs_fair_phase_swap_8.json) preserve every result.

**Interpretation:** Using fair-value choices outside `PLAY_TURN` improves this model substantially over using it everywhere, yet still trails full fair-value. Letting the model control only openings also underperforms full fair-value, especially against the strong mix. The hybrids change several phase decisions at once, and eight boards leave wide intervals; these results do **not** identify a single causal defect or justify promoting a hybrid. They support focused work on openings, robber/discard decisions, and turn strategy with larger held-out tests.

## Reproduce

Use `PYTHONHASHSEED=0`, eight boards, seed `20269300`, and the same 20-second game cap for every run:

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/mixed_eval.py --agent-file catanLabs/Competitive/opening_rescue_agent.py --checkpoint catanLabs/Competitive/configs/phase_swap_model_turn.json --boards 8 --seed 20269300 --max-game-seconds 20 --report catanLabs/Competitive/runs/phase_swap_model_turn_8.json
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/mixed_eval.py --agent-file catanLabs/Competitive/opening_rescue_agent.py --checkpoint catanLabs/Competitive/configs/phase_swap_model_opening.json --boards 8 --seed 20269300 --max-game-seconds 20 --report catanLabs/Competitive/runs/phase_swap_model_opening_8.json
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/mixed_eval.py --agent-file catanLabs/Competitive/fair_value_agent.py --checkpoint catanLabs/Competitive/configs/fair_army_finish.json --boards 8 --seed 20269300 --max-game-seconds 20 --report catanLabs/Competitive/runs/phase_swap_fair_control_8.json
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/mixed_eval.py --agent-file catanLabs/Transformer/legacy_v2_agent.py --checkpoint catanLabs/Transformer/runs/gate3_matched/transformer/best.pt --boards 8 --seed 20269300 --max-game-seconds 20 --report catanLabs/Competitive/runs/phase_swap_full_transformer_8.json
```

Run [`compare.py`](compare.py) on any two reports for the paired board-bootstrap interval. The six saved comparisons have the suffix `_phase_swap_8.json` in `Competitive/runs/`.
