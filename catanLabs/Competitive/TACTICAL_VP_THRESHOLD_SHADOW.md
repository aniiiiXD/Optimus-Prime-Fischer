# Tactical VP threshold: paired shadow probe

**Status: research only; FairValue stays incumbent.** The frozen
[`TacticalVPAgent`](tactical_vp_agent.py) normally scans deterministic
two-action VP lines after rolling only from 7 actual VP. This probe asks how
many additional lines it finds from 5 VP and from the start of the game.
FairValue made **every actual move**, so none of these figures measure wins.

The [two-board raw report](runs/tactical_vp_threshold_shadow_2boards.json)
uses fresh seeds **20274000–20274001**, all four candidate seats, Value plus
two AlphaBeta1 opponents, and the same provisional Catanatron v5 rules.
All eight games completed without caps or errors. The probe scanned each
post-roll, multi-action FairValue position once with `tactical_min_vp=0`,
then filtered those same decisions by actual VP for the 5 and 7 views. The
minimum VP parameter changes only eligibility; a focused test verifies
identical tactical results where two thresholds overlap. This makes the
three views exactly matched without tripling scan CPU.

| Minimum actual VP | Eligible decisions | Exact VP lines found | Proposed overrides | Games with override | Distinct turns with override | Scanner p95 / max |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 49 | 10 | 5 | 3/8 | 5 | 20.8 / 34.1 ms |
| 5 | 79 | 25 | 11 | 5/8 | 10 | 17.1 / 34.1 ms |
| 0 | 189 | 61 | 27 | 8/8 | 25 | 9.9 / 34.1 ms |

**The 5-VP gate adds six proposed changes at 5–6 VP**, across four games and
five distinct turns. Four changed which maritime trade to make; two changed
`END_TURN` to a trade that unlocked a road or city point. All six searched
second actions remained legal and were selected by the tactical policy in
their copied follow-up states. The 0-VP gate adds 16 early proposed changes,
mostly alternative trades, across all eight games.

The extra 5–6 VP scans took **7.5 ms p95, 21.4 ms max**; 0–4 VP scans took
**3.2 ms p95, 5.9 ms max**. The lower overall p95 at threshold 0 reflects
many cheap early positions; it is not a faster policy. There were **zero
root/follow-up budget truncations** at any threshold. Every proposed root was
in the real legal list and validated on a copied game. Nine hidden-hand,
deck, and RNG perturbations (three per VP band) preserved FairValue and
scanner choices. Shadow scanning did not advance the live game RNG.

One of the five late proposals selected a different continuation when the
tactical policy decided again in the copied state. The new 5–6 VP proposals
all kept their searched continuation. A legal immediate point can still
sacrifice production, resources, or board access, so additional proposals
are **opportunities to test**, not evidence of stronger play. A next bounded
gate is a fresh matched all-seat screen of the 5-VP candidate against both
FairValue and the frozen 7-VP candidate, with both opponent mixes, completed
games, paired board intervals, and latency. Do not promote from this shadow.

Reproduce the probe:

```bash
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/tactical_vp_threshold_probe.py --boards 2 --seed 20274000 --max-game-seconds 20 --report catanLabs/Competitive/runs/tactical_vp_threshold_shadow_2boards.json
```
