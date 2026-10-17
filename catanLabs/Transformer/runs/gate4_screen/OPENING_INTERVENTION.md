# Gate 4 diagnostic: replace opening placements only

The same frozen Gate 3 Transformer was screened on fresh seeds
**20271100–20271107**, all four seats and both mixed opponent compositions.
The intervention used our fair-value policy only for the initial settlement and
road prompts; the Transformer made every later choice. The
[control](transformer_opening_control.json) and
[intervention](opening_rescue.json) each completed 64 games with zero errors
or caps. This hybrid is a research probe, not a promoted submission agent.

| Mix | Transformer control | Fair openings + Transformer | Mean VP control → intervention |
|---|---:|---:|---:|
| Diverse | 1/32 wins | 2/32 wins | 4.16 → 5.19 |
| Strong | 3/32 wins | 3/32 wins | 4.19 → 5.00 |

The [paired board comparison](opening_rescue_vs_control.json) estimates a
diverse-mix win difference of +3.1 percentage points (95% board-bootstrap
interval −6.2 to +12.5) and a strong-mix difference of 0.0 points (−15.6 to
+15.6). These eight-board screens cannot establish a win-rate gain. The VP
increase supports an opening effect, but the learned midgame remains weak.

The [on-policy trace](../gate4_onpolicy/diagnostic.md) found 0/8 exact fair
teacher matches for opening settlement choices in four additional games.
Architecture replay found a concentration on node IDs 8 and 18 across native
opening records; it is investigating whether that is a learned shortcut or an
input defect. Do not attribute all losses to openings from this experiment.
