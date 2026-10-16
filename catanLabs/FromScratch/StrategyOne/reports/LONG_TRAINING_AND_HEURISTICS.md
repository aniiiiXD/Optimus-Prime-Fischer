# Isolated longer-training and heuristic experiments

Branch: `experiment/catan-longer-training-heuristics`  
Worktree: `/Users/anixd/Documents/Hyde-catan-experiments`

The baseline code was copied and committed in this branch. Dataset and base
checkpoints are read through local symlinks to the original workspace. New
checkpoints and reports are written only to the experiment worktree.

## Arms

All arms start with the same calibrated MLP. The current settlement target
head is its four-epoch checkpoint.

| Arm | Single change |
| --- | --- |
| Current | Four-epoch learned settlement head |
| Longer 7 | Same head architecture, trained for seven epochs |
| Longer 12 | Same head architecture, trained for twelve epochs |
| Settlement pips | Current head, then override settlement location with the highest adjacent dice-pip count |
| Buy Dev at end | Current head, then buy a legal development card if the model chose End Turn while holding at least eight resource cards |

The two heuristics were evaluated separately. Both use only visible game state
and legal actions.

## Target classification during training

| Epoch | Held-out settlement-location match | Training loss |
| ---: | ---: | ---: |
| 4 | 39.7% | 2.102 |
| 7 | **42.1%** | 1.798 |
| 12 | 38.1% | 1.408 |

Training loss kept falling after epoch 7 while held-out target matching fell.

## Two fresh matched 128-game screens

Seeds `20266900`–`20267027` and `20267100`–`20267227`. Each arm played the
same seed, opponent, and seat schedule under the RNG-isolated Catanatron
contract. None of these games was used in training.

| Arm | Screen A | Screen B | Combined | Value Base | Value Contender |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current | 62/128 | 63/128 | **125/256 (48.8%)** | 4/52 | 3/52 |
| Longer 7 | 61/128 | 64/128 | **125/256 (48.8%)** | 6/52 | 3/52 |
| Longer 12 | 57/128 | 66/128 | **123/256 (48.0%)** | 4/52 | 3/52 |
| Settlement pips | 60/128 | 58/128 | **118/256 (46.1%)** | 3/52 | 7/52 |
| Buy Dev at end | 64/128 | 63/128 | **127/256 (49.6%)** | 6/52 | 6/52 |

Against the same current-model games, longer 7 flipped 22 losses to wins and
22 wins to losses. Longer 12 flipped 21 to wins and 23 to losses. The pips rule
flipped 20 to wins and 27 to losses. The Buy Dev rule flipped 11 to wins and 9
to losses; it changed 1,105 individual decisions across 256 games.

## Verdict

Longer target-head training improves teacher-action matching up to epoch 7,
but it does not improve overall win rate here. The simple dice-pip settlement
rule makes overall results worse in both screens. The Buy Dev rule gives a
small two-win overall increase and slightly better value-bot counts, but the
paired result is too small to justify adopting it as the agent's default.

All arms remain on the feature branch. The original workspace and current
agent checkpoints were not modified by this experiment.
