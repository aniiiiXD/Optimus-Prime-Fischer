# Gate 0 handoff: visible v2 pipeline

**Date:** 2026-10-15. **Local target:** the provisional four-player
Catanatron v5 [rules manifest](../rules_manifest.json). The official event
rules are unknown until the event day.

## What changed

- [VisibleObservation](visible_observation.py) is a frozen primitive-only
  snapshot. `VisibleAgentAdapter` passes only that snapshot to new policies
  and maps their legal-action index back to the engine action.
- [The encoder](model.py) consumes the snapshot, gives ORE ports a distinct
  feature bit, and identifies its output as feature schema v2. Dataset and
  checkpoint loaders reject historical v1 artifacts.
- [The shared v5 protocol](../Catanatron/tournament_protocol.py) isolates
  copied RNG streams, honors hypothetical roll/card outcomes, and sorts legal
  actions. [Collection](collect.py) defaults to our fair-value teacher;
  AlphaBeta1 is explicit and marked as an unverified full-game teacher.

## Checks run

| Check | Result |
|---|---|
| `start/.venv/bin/python -m unittest catanLabs.Transformer.test_collect_protocol catanLabs.Transformer.test_visible_observation catanLabs.Transformer.tests.test_visible_encoder catanLabs.Transformer.tests.test_feature_schema -v` | 13/13 passed. Includes deterministic replay, all-seat completed games, hidden hand/deck/RNG perturbation, legal adapter, ORE bit, and v1 rejection. |
| `start/.venv/bin/python catanLabs/Competitive/check_visibility.py --agent fair` | 112 decisions invariant to tested hidden-state changes. |
| 100-game fair-teacher collection, seeds 20266000–20266099, 150 s wall cap | [Manifest](runs/gate1_profile/dataset.json): 100/100 completed, 29,676 unique v2 encoded decisions, 55.288 s wall time, zero recorded illegal actions. Dataset SHA-256 `0716a25b8883aa32179e65009f885ddfe9c3a5dbfcdeaf4ee4bb4c15907e436d`. |
| Manual encoded-record inspection | Sampled rows 0, 1, 700, 800, 1000, 10000, 20000, and 29675: labels were in legal range, action type/group/target were plausible, and ORE port bits appeared in node tokens. All 29,676 labels satisfy `0 <= label < candidate_counts`. |
| Short CPU update, 20 s requested optimizer cap | [Training report](runs/gate1_profile/short_update/training.json): 13,857 parameters, 2 epochs, 21.253 s measured optimizer loop including validation, 35.38% held-out teacher agreement. This was a timing/plumbing probe. |
| Four-seat benchmark smoke versus 3× Value | [Report](runs/gate1_profile/agent_smoke.json): four completed games, zero illegal actions/errors, 0/4 wins, 3.25 mean final VP, 0.842 ms warm p95 decision latency over 309 samples. Four games do not estimate playing strength. |

## Current boundary

Gate 0's visible-only model path is operational. The 100-game dataset stores
encoded arrays only; the Gate 2 raw-snapshot/reconstruction and sealed
board-level split requirements are still in progress. The 20-second checkpoint
is a diagnostic artifact and must not replace the fair-value incumbent.

**Next gate:** finish the phase-specific failure and optimizer profiling report,
then collect a raw, audited 100-game v2 dataset and use it for a same-data MLP
versus Transformer comparison.
