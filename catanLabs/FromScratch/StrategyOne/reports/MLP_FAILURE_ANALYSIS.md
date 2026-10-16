# MLP loss diagnosis — v1

**Scope:** diagnosis only. This did not retrain the MLP or change its policy.

## Measurement

- 50 complete 1v1 games: frozen MLP versus Catanatron `F_BASE`.
- MLP seat alternated; games used fixed seeds `20264001` through `20264050`.
- At every MLP decision, `F_BASE` was queried on an isolated game copy only to record its preferred legal action. It never changed the live game.
- Result: **2 / 50 wins (4%)**.

## What is going wrong

| Area | Evidence | Reading |
| --- | --- | --- |
| Development cards | In 435 decisions, MLP chose `END_TURN` while `F_BASE` chose `BUY_DEVELOPMENT_CARD`. In the 20% data check, the MLP got **0 / 173** teacher-labelled development-card purchases correct (169 were predicted as `END_TURN`). | This is the biggest identifiable decision failure. The model leaves usable resources idle instead of converting them into value. |
| Resource conversion | By late game, MLP held 3.76 more resource cards on average, yet had 3.11 fewer VP. Across lost games it finished with 4.19 more cards. | The failure is conversion, not simply collecting resources. |
| Production and expansion | From midgame onward, the MLP had lower expected production: -10.39 dice pips in midgame and -22.84 late game. Late game it also trailed by 3.34 roads and 1.40 settlements. | It does not turn early board position into enough production and reachable expansion. |
| Territory target choice | Both players selected a territory-building action type frequently, but the exact legal action agreed only 4.0% of the time. | The MLP often picks a different road or settlement location even when it knows to build. |
| Robber target choice | Exact agreement on robber actions was 0.0%. | Robber placement/victim selection is unlearned in the current model. This report does not yet quantify its separate impact on wins. |
| City timing | The MLP chose `BUILD_CITY` when `F_BASE` chose `END_TURN` 28 times and `BUY_DEVELOPMENT_CARD` 20 times. | City building is being selected at moments where the stronger player prefers another conversion. |

## Why development cards are the first model problem to inspect

Only 1,758 of 156,471 dataset labels (1.12%) are `BUY_DEVELOPMENT_CARD`; the current ranker never selected that action in the 20% check. This is a concrete coverage/class-imbalance and ranking failure, not evidence that the game needs a hand-written heuristic.

## Phase trend

| Phase | MLP VP minus `F_BASE` | MLP production-pips minus `F_BASE` | MLP resource cards minus `F_BASE` |
| --- | ---: | ---: | ---: |
| Opening | -0.05 | -1.80 | +0.64 |
| Midgame | -0.91 | -10.39 | +2.00 |
| Late game | -3.11 | -22.84 | +3.76 |

The opening is close. The compounding loss starts in the midgame.

## Diagnostic artifact

The complete machine-readable result is in [mlp_loss_diagnosis_v1.json](mlp_loss_diagnosis_v1.json). The reproducible diagnostic is `../diagnose_losses.py`.
