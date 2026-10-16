# Sealed 128-game PPO verdict

## Protocol

- 128 fresh Catanatron games, seeds `20265128` through `20265255`.
- Seeds were not used in dataset collection, PPO training, earlier diagnostics, or earlier evaluation.
- Opponents cycled in a fixed order; seats alternated.
- The imitation MLP and PPO MLP played the identical seed/opponent/seat schedule.
- No model trained on these games.

## Result

| Opponent | Imitation MLP | PPO MLP |
| --- | ---: | ---: |
| Value Base | 1 / 26 | 2 / 26 |
| Value Contender | 0 / 26 | 1 / 26 |
| Victory Point | 15 / 26 | 14 / 26 |
| Weighted Random | 12 / 25 | 11 / 25 |
| Random | 17 / 25 | 16 / 25 |
| **Total** | **45 / 128 (35.2%)** | **44 / 128 (34.4%)** |

## Verdict

**No: this PPO configuration is not working yet.** It is one win below the original MLP on a fresh, matched test. The isolated gains against the two value bots are too small to outweigh losses against the other opponents.

The terminal-reward-only PPO run did not repair the known resource-conversion failure. Do not extend the same setup blindly. The next RL experiment needs an imitation-learning anchor during PPO and an offline-pretrained critic, while preserving a newly generated final holdout screen for model selection.
