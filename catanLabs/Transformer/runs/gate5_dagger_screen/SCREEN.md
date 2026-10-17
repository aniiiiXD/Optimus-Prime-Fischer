# On-policy fair-label fine-tune: held-out screen

**Decision: no promotion.** The on-policy fine-tune improved average points a
little but remains far below the fair-value incumbent on provisional
four-player Catanatron v5.

The [supplemental corpus](../../ON_POLICY.md) contains 60,000
unique decisions from 98 complete self-play games, all labeled by the fair
teacher before the behavior move. The [merged audit](../gate4_merged_v3/dataset.audit.json)
passed on 89,899 rows with no exact duplicates or detected hidden-field or
split errors. The original 10 final games remain sealed. Fine-tuning began
from [the v3 parent](../gate3_v3_choice/best.pt), used 46,414 train decisions
with a choice and a fresh AdamW optimizer at learning rate 1e-4, and stopped
at 592.9 wall seconds on CPU. Including the parent's 274.1-second run, this
lineage used 867.0 training wall seconds across two capped runs.

All three candidates played fresh seeds **20271800–20271807** in every seat
against the same two opponent mixes. All 192 games completed without caps or
errors. These seeds are outside training, validation, sealed final, and
earlier screens.

| Candidate | Diverse wins / 32 | Mean VP | Strong wins / 32 | Mean VP | Warm p95 latency |
|---|---:|---:|---:|---:|---:|
| [Fair-value incumbent](fair.json) | 12 | 7.78 | 9 | 7.09 | 3.77–4.25 ms |
| [V3 parent](parent_v3.json) | 1 | 3.97 | 0 | 3.28 | 0.92–0.98 ms |
| [On-policy fine-tune](finetuned_v3.json) | 1 | 4.41 | 1 | 3.59 | 0.89–0.95 ms |

The [paired fine-tune versus fair report](finetuned_vs_fair.json) gives
win-rate differences of −34.4 percentage points in the diverse mix
(board-bootstrap 95% interval −56.2 to −18.8) and −25.0 points in the strong
mix (−40.6 to −9.4). The [paired parent comparison](finetuned_vs_parent.json)
shows 0.0 and +3.1 points, with intervals including zero. The mean VP gains
are +0.44 and +0.31. Eight boards are a screen, but these results clearly do
not justify promoting this checkpoint or running a 100-board confirmation.

The merged validation accuracy was 74.9%, but its ten original base
validation games already influenced parent checkpoint selection. It is useful
for early stopping, not a clean model-to-model improvement estimate. The
fresh matched games above decide promotion. This experiment shows that more
fair-teacher labels on self-play states did not close the observed game gap;
it does not establish that all forms of on-policy learning or reinforcement
learning will fail.
