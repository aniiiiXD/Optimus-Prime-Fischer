# Catan-family adaptation map

The event rules arrive on the day. Our primary measured results apply to the
four-player Catanatron v5 rules in [rules_manifest.json](rules_manifest.json).
Use this map to turn the event specification into concrete interface changes
before any training or win-rate claim. These are code-inspection findings, not
measured transfer results.

| Event rule difference | Present local behavior | Required adaptation |
|---|---|---|
| Same base board, different target VP, discard limit, robber rule, or number placement | The visible snapshot and encoder carry these values. The evaluator currently fixes them to v5 defaults. | Parameterize game creation and run legal-action, hidden-information, and matched tournament smokes. Retrain or recalibrate if results fall. |
| Three players | The observation builder can freeze the seated players, but the encoder requires exactly four player tokens. | Version a three-player encoder and checkpoint; never pad or silently reuse four-player weights without a measured compatibility test. |
| Mini or expanded board | The encoder requires 19 tiles, 54 nodes, 72 edges, and 150 state tokens. | Version the topology representation, tensor shapes, dataset, and checkpoint. |
| Domestic trade offers/responses | The current benchmark excludes them. Some action names exist in the vocabulary, but collection and evaluation do not cover them. | Audit action serialization and legal-action uniqueness, collect trade decisions, and test full games with trades enabled. |
| Different observation or legal-action API | The visible adapter and action descriptors depend on Catanatron field and action meanings. | Add a contract-specific adapter with round-trip legal-action tests and a new schema; keep hidden fields outside the model. |
| Different per-move CPU or memory limit | We measure warm local latency; event limits remain unknown. | Measure cold start, model load, snapshot encoding, and full decision time on the target CPU before packaging. |
| Different scoring or opponent distribution | Current promotion uses four-player wins against two fixed local mixes. | Freeze new held-out seeds and seats; update the promotion test to match the actual scoring contract. |

On the event day, first fill the `competition_contract` section of the rules
manifest and write a short diff against `local_rules`. Then run four-seat
legality and runtime smokes, adapt data and checkpoints only where the diff
requires, and compare candidates on fresh event-rule boards. A local v5 win
rate does not transfer automatically to another family.

## TacticalVP scalar-rule drill, 16 October 2026

The promoted `TacticalVPAgent` and its FairValue control were tested on fresh
eight-board, all-seat screens against Value plus two AlphaBeta1 opponents in
each supported variant. Every candidate and control game completed legally;
this is a useful compatibility check, but eight boards are too few to make a
transfer-strength claim. Win differences are paired tactical minus FairValue,
with board-clustered 95% bootstrap intervals.

| Variant | Tactical / Fair wins | Difference | 95% interval | Reading |
|---|---:|---:|---:|---|
| Target 8 VP | 9/32 vs 9/32 | 0.0 points | −12.5 to +15.6 | Inconclusive |
| Target 12 VP | 19/32 vs 18/32 | +3.1 points | 0.0 to +9.4 | Inconclusive |
| Discard at 5 cards | 14/32 vs 15/32 | −3.1 points | −9.4 to 0.0 | Possible regression; recalibrate before use |
| Friendly robber | 17/32 vs 12/32 | +15.6 points | +3.1 to +28.1 | Promising, needs independent confirmation |
| Random number placement | 12/32 vs 10/32 | +6.3 points | 0.0 to +15.6 | Inconclusive |

The [family evaluator](Competitive/family_eval.py) parameterizes exactly these
five Catanatron rules, stores per-game data, and checks legal actions. The
[paired comparator](Competitive/family_compare.py) validates the seed, seat,
opponent, and rule match before computing intervals. Raw [candidate](Competitive/runs/tactical_vp_min0_family_screen_8.json),
[control](Competitive/runs/fair_family_screen_8.json), and [comparison](Competitive/runs/tactical_vp_min0_vs_fair_family_screen_8.json)
reports are retained. Player-count, topology, domestic-trade, and API changes
remain unsupported until their adapters and data schemas are versioned.
