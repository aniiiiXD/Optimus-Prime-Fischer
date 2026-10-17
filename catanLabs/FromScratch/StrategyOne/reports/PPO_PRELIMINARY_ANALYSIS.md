# PPO preliminary run — result

## Training

- 8 PPO updates × 8 complete games = **64 games**.
- About **21,390 policy decisions**.
- Policy initialized from the imitation MLP.
- Opponent mixture: frozen policy snapshots, Value Base, Victory Point, Weighted Random, and Random.
- Reward: terminal game result only (`+1` win, `-1` loss). No hand-written reward shaping.

## Fixed 100-game evaluation

Both models were evaluated with the same 20 seeded games against each opponent.

| Opponent | Imitation MLP | PPO MLP | Change |
| --- | ---: | ---: | ---: |
| Value Base | 2 / 20 | 3 / 20 | +1 |
| Value Contender | 0 / 20 | 0 / 20 | 0 |
| Victory Point | 14 / 20 | 15 / 20 | +1 |
| Weighted Random | 10 / 20 | 10 / 20 | 0 |
| Random | 13 / 20 | 11 / 20 | -2 |
| **Total** | **39 / 100** | **39 / 100** | **0** |

There is no reliable win-rate improvement from this preliminary PPO run.

## Failure diagnosis versus Value Base

On the same 50 diagnostic seeds, the imitation MLP won 2 games while the PPO MLP won 0.

| Signal | Imitation MLP | PPO MLP |
| --- | ---: | ---: |
| `END_TURN` when Value Base chose `BUY_DEVELOPMENT_CARD` | 435 | 431 |
| Late-game production-pip gap | -22.84 | -23.01 |
| Late-game VP gap | -3.11 | -3.16 |
| Resource-card gap in final losses | +4.19 | +5.80 |
| Exact robber-action agreement | 0.0% | 0.3% |

PPO did not learn the development-card conversion failure. It still hoards resources and loses production/expansion from the midgame.

## Interpretation

This is expected from a small, terminal-reward-only run. A win/loss signal arrives hundreds of decisions after the choice that mattered, while the rare development-card action represents only 1.12% of the imitation dataset. Running much more of this exact PPO setup blindly is unlikely to fix that bottleneck efficiently.

The next PPO version should retain an imitation anchor during policy updates and train the critic offline on recorded terminal outcomes before policy fine-tuning. Those are learning signals, not game-play heuristics.
