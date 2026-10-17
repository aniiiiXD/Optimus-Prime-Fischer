# Learned calibration-layer results

## Change made

The original legal-action MLP is frozen. A learned adapter was trained above it:

1. A state-conditioned residual score for every action type.
2. A learned state-conditioned adjustment to the `END_TURN` score.

The adapter starts at zero, so it initially behaves exactly like the existing MLP. Training used external Catanatron teacher labels and a balanced pairwise loss whenever the teacher chose `BUY_DEVELOPMENT_CARD` while `END_TURN` was legal. No hand-written Catan decision rule or reward was added.

## Supervised target

On the 20% game-level validation split:

| Metric | Original MLP | Calibrated MLP |
| --- | ---: | ---: |
| Buy Development exact accuracy | 0 / 173 (0.0%) | 78 / 173 (45.1%) |
| Full legal action accuracy | 75.8%* | 74.5% |
| Non-forced action accuracy | 49.9%* | 47.3% |

`*` The older MLP had partially overlapping evaluation rows because it was trained under an earlier 90/10 split. The calibration adapter itself did not train on the new 20% validation games.

## Fresh 128-game win evaluation

The two models played identical new seeds, opponents, and alternating seats. The evaluator isolates Catanatron's search RNG so hypothetical search cannot alter the live dice/card sequence. None of these games were used for dataset collection or calibration training.

| Opponent | Original MLP | Calibrated MLP |
| --- | ---: | ---: |
| Value Base | 1 / 26 | 3 / 26 |
| Value Contender | 1 / 26 | 2 / 26 |
| Victory Point | 11 / 26 | 15 / 26 |
| Weighted Random | 10 / 25 | 18 / 25 |
| Random | 14 / 25 | 19 / 25 |
| **Total** | **37 / 128 (28.9%)** | **57 / 128 (44.5%)** |

## Value Base diagnosis

On the same 26 fresh Value Base seeds:

| Signal | Original MLP | Calibrated MLP |
| --- | ---: | ---: |
| Wins | 1 / 26 | 3 / 26 |
| `END_TURN` when teacher chose Buy Development | 223 | 152 |

The adapter changes the measured action mix and the matched game results. Separate Value Base loss details are recorded in `VALUE_BASE_LOSS_AUDIT_V5.md`.
