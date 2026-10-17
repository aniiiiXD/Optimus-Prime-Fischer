# V3 public-count and choice-only training screen

**Decision: no promotion.** The v3 policy remains far below the fair-value
incumbent in the provisional four-player Catanatron v5 benchmark. V3 adds
public opponent card counts to the encoder; its training run also skipped
one-action rows, which have zero masked cross-entropy gradient. This screen
cannot attribute differences to either change alone.

The v3 Transformer has 13,857 parameters and used 274.1 wall seconds on CPU
under the 600-second cap. Early stopping retained epoch 20 of 26. It trained
on 13,394 decisions with a choice and excluded 10,538 forced rows. Its exact
fair-teacher agreement on the unchanged 3,060-row validation games was 71.2%,
versus 77.6% for frozen v2.

All agents played seeds **20271400–20271407**, each in all four seats, against
the same `mixed_diverse` and `mixed_strong` assignments. These boards are
outside the collection, validation, sealed final, and previous screen seeds.
All 192 games completed, with zero caps or errors.

| Candidate | Diverse wins / 32 | Mean VP | Strong wins / 32 | Mean VP | Warm p95 decision time |
|---|---:|---:|---:|---:|---:|
| [Fair-value incumbent](fair.json) | 10 | 7.06 | 11 | 6.94 | 3.18–3.68 ms |
| [Frozen v2 Transformer](v2_transformer.json) | 1 | 4.38 | 1 | 4.12 | 0.78 ms |
| [V3 choice-only Transformer](v3_choice_transformer.json) | 1 | 4.06 | 1 | 4.06 | 0.82–0.85 ms |

The [paired comparison with fair](v3_vs_fair.json) gives v3 win-rate
differences of **−28.1 percentage points** in the diverse mix (board-bootstrap
95% interval −43.8 to −12.5) and **−31.2 points** in the strong mix (−43.8 to
−15.6). The [paired v3–v2 comparison](v3_vs_v2.json) is 0.0 points in each
mix, with intervals −9.4 to +9.4. Eight boards support a rejection of this
checkpoint as an upgrade, not a fine ranking of the two weak models.

Next: collect fair-teacher labels on complete games reached by this v3 policy,
merge them with the original training partition while leaving original
validation/final boards sealed, and test one bounded CPU retrain. No
competition-strength claim follows from teacher agreement alone.
