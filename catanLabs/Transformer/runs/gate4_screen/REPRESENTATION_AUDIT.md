# Gate 4 representation audit

The v2 Transformer won 0/32 games against the strong mix, despite 77.6% overall teacher-action agreement. The validation number includes forced decisions: agreement on decisions with multiple legal actions was 60.2%. Initial-settlement agreement was 20.0%, and robber agreement was 16.4% ([validation report](../gate3_matched/model_comparison.md), [screen report](SCREEN.md)).

## Confirmed findings

- `VisiblePlayer` retains each opponent's public `resource_count` and `development_card_count` (`visible_observation.py`), but `encode_visible_observation` never writes them into player tokens (`model.py`). Replacing one opponent's counts from 0/0 to 7/3 leaves all seven encoded arrays bitwise identical. This loses public information useful for robber, victim, Monopoly, and threat decisions. It does not by itself explain weak opening placements.
- Across all 800 initial-settlement rows in the v2 dataset, the model selected node 8 or 18 in 575 rows (71.9%). The teacher chose those nodes in 164 rows (20.5%). Exact match was 23.75% on first placements and 26.25% on second placements. Mean encoded production at the selected node was similar to the teacher: first placement 0.3062 versus 0.3091; second 0.2748 versus 0.2747. The model's opening problem therefore needs more than a simple production feature check.
- In all 29,899 v2 records, legal-action descriptors had no within-state collisions and target IDs stayed in range. A 64-record raw-to-NPZ sample re-encoded exactly. Changing legal-action order in a 54-action opening changed logits by at most 1.43e-6 and preserved the selected action. The raw snapshot keeps own private cards and public opponent fields; opponent private cards, deck, and RNG are outside it.
- Public victory points matched the engine's public building/award totals in a 3,000-record raw sample. Own hidden victory-point cards remain in the acting player's private development-card vector. The snapshot's phase and legal-action sets matched records in that sample.

## Limits and next experiment

Node production and port are encoded, so the opening collapse is a measured policy failure, not a proven action-mapping bug. The encoder does not explicitly pass tile-to-node and edge-to-node incidence to attention; it mainly relies on fixed entity IDs to learn board relationships. This may weaken the small one-layer policy, but requires an ablation.

Version the feature schema, dataset, and checkpoint to v3. Add public `resource_count` and `development_card_count` to player-token columns 32 and 33. Re-encode the stored immutable v2 raw observations with the v3 encoder into a new NPZ; preserve original labels, game/board split membership, and historical v2 artifacts. Recompute dataset and split hashes. Train a fresh checkpoint and compare it with v2 on held-out games. Separately test opening-focused sampling or candidate-local graph features; the count fix should not be credited for any opening gain without this separation.
