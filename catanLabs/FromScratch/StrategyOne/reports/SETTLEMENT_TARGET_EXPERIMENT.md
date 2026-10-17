# Settlement target classifier experiment

## Change

The calibrated MLP still decides whether to build a settlement. A small
learned head reranks legal settlement **locations** only when the calibrated
MLP already chose `BUILD_SETTLEMENT` and at least two target nodes are legal.
It receives local board relationships: adjacent terrain and dice numbers,
ports, nearby roads and buildings, current production, own resources, and VP.
It has no node-ID weight and no manually weighted Catan score.

Training used 2,396 teacher-labelled settlement decisions with multiple
locations. Validation used 546 decisions from distinct game IDs, excluded
from this head's training.

## Held-out target matching

| Situation | Calibrated MLP | With target head |
| --- | ---: | ---: |
| All settlement choices (546) | 26.2% | 39.7% |
| Opening placements (412) | 13.4% | 28.6% |
| Later placements (134) | 65.7% | 73.9% |

The frozen calibrated MLP was trained separately; some of its original
training games overlap these head-validation games. The independent full-game
screens below are the clean win-rate comparison.

## Fresh matched games

Both models played each seed, opponent, and seat under the same RNG-isolated
Catanatron contract. The target head was not trained on either screen.

| Opponent | Calibrated MLP | With target head | Change |
| --- | ---: | ---: | ---: |
| Value Base | 3 / 52 | 3 / 52 | 0 |
| Value Contender | 2 / 52 | 3 / 52 | +1 |
| Victory Point | 39 / 52 | 40 / 52 | +1 |
| Weighted Random | 32 / 50 | 39 / 50 | +7 |
| Random | 37 / 50 | 41 / 50 | +4 |
| **Total** | **113 / 256 (44.1%)** | **126 / 256 (49.2%)** | **+13 wins** |

Screen A: 59 / 128 → 64 / 128. Screen B: 54 / 128 → 62 / 128.
Across screen A, the classifier reranked 311 settlement decisions and changed
the chosen node 203 times.

## Reading

This is evidence that explicit local relationships help settlement-location
selection. The matched win gain repeated across two seed ranges. The value
bots remain much stronger; this experiment does not show that their gap is
solved. Road and robber targets were not changed.
