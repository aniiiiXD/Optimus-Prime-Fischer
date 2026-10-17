# Learnable OneVOne: completed redesign and tests

Branch: `experiment/onevone-learnable`. Final playing source is frozen at
`66e2b06`; later commits record completed results and documentation.

The redesigned policy runs with a trained MLP and supports a trained transformer
through the same interface. Rules and legal actions come from the original
engine. The existing two-action tactical preference is a soft expert prior;
bounded guaranteed-win protection remains separate. The learning contribution
can be adjusted or the strategic prior disabled in configuration.

**The redesign is operational. The trained learner has not established a
strength improvement and is not promoted over the original OneVOne.**

## Final 200-game test and controls

Official kit `04fffac90e00610f60e0b06d919f262f4b3ed9db`, Catanatron
`d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`, official CPU clock enforced.
Each row contains 100 boards, played from both seats. The first three use the
same board/dice/seat schedule, seeds **2026890000–2026890099**, against official F.

| Policy | Wins against F | Mean CPU/game | Worst CPU/game | Errors / overruns |
|---|---:|---:|---:|---:|
| Original OneVOne | 110/200 (55.0%) | 185.6 ms | 636.4 ms | 0 / 0 |
| Redesigned expert, learning disabled | 114/200 (57.0%) | 221.7 ms | 734.9 ms | 0 / 0 |
| Redesigned expert + trained MLP | **112/200 (56.0%)** | 225.3 ms | 784.2 ms | **0 / 0** |

On another 100 fresh boards, seeds **2026891000–2026891099**, the trained MLP
hybrid won **86/200 (43.0%) directly against original OneVOne**. Its mean/worst
CPU was 204.9/720.2 ms; neither side had errors or overruns.

All four final runs finished without illegal indices, exceptions, CPU overruns,
post-limit random moves, timeouts, crashes or draws on either side. Timings are
local Mac measurements, with the unchanged official clock enforced.

| Paired comparison against F | Difference | Board-bootstrap 95% interval | Gained / lost games |
|---|---:|---:|---:|
| Learned hybrid minus original | +1.0 pp | [-7.5, +9.5] pp | 42 / 40 |
| Learned hybrid minus same redesigned expert | -1.0 pp | [-9.5, +8.0] pp | 40 / 42 |
| Redesigned expert minus original | +2.0 pp | [+0.5, +4.0] pp | 4 / 0 |

Intervals resample whole two-seat board pairs 10,000 times. The expert's four
extra wins are a small observed improvement needing independent confirmation.
The learned model changes many outcomes but shows no reliable gain. Its direct
match against OneVOne also gives no reason to promote it. The architecture
supports learning; useful prediction loss alone does not establish better play.

[Comparison, freeze and split checks](reports/final_v2_comparison.json),
[learned games](reports/final_v2_learned_f_200.json),
[original control](reports/final_v2_original_f_200.json),
[same-expert control](reports/final_v2_heuristic_f_200.json),
[direct match](reports/final_v2_head_to_head_200.json).

## Training and actual model activity

Fresh final training data: **512 complete games on 256 boards**, plus
**423 continuation games** from alternative action choices; **10,354 examples**.
Generation took 50.5 seconds. The policy mixed small amounts of exploration
with the expert and played against F. Labels are actual completed-game wins,
including soft win frequencies from repeated continuations. These simulations
were unclocked; the final evaluations above were clocked.

Whole boards and both seats remain together: 205 training boards / 51 validation
boards, 8,294 / 2,060 examples. Training seeds are 2026871000–2026871255, disjoint
from final evaluation seeds. Continuation siblings remain in their parent
board's split. Generation checks that its source stays unchanged.

| Model | Parameters | Selected epoch / 15 | Held-out outcome BCE | Max Torch/NumPy logit difference |
|---|---:|---:|---:|---:|
| MLP | 17,217 | 6 | 0.5205 | 9.54e-7 |
| Transformer | 13,531 | 2 | 0.5436 | 5.90e-7 |

The MLP was selected by lower validation BCE **before** final games. The blend
was fixed at prior weight 1 and learning weight 0.35. No final board was used
to select architecture, epoch, checkpoint or blend. Training used the local
PyTorch 2.7.1 environment; actual play uses NumPy and the pinned official engine.

In eight separate unclocked activity-audit games, the final MLP was called on
368 decisions and changed **42 actual executed choices**, after tactical
protection. There were five winning-line guards, zero model failures and zero
clock fallbacks. The report includes changed-action examples. This audit proves
model influence, not playing strength.

The final MLP passed a 20-game official smoke test (10/20); the final transformer
passed a four-game official smoke test (2/4). Both had zero faults or overruns.
The transformer was not given a 200-game strength evaluation in this redesign.

[Data manifest](reports/data_manifest_v2.json),
[MLP training](models/mlp_v2/training.json),
[transformer training](models/transformer_v2/training.json),
[frozen selection](reports/selection_v2.json),
[activity audit](reports/model_activity_v2.json).

## Behavioral verification

Nine tests pass. They check that every legal move is exposed and hypothetical
scoring preserves the input; changing opponent private development identities,
private VP or future deck order does not change visible features/prior scores;
a learner can overturn the expert ordering; immediate and trade-then-city wins
are protected; ordinary tactical preferences remain overridable; invalid model
outputs fall back visibly; low clock reserve returns a legal move; and a missing
configured checkpoint fails at load time.

Known tactical gains are lower bounds from a bounded search. The guard is
incomplete and does not prove that no winning line exists when it finds none.
The observation encoder never reads opponent private VP or development identity.
Development buys retain unknown outcome features; robber moves retain expected
theft utility without sampling the stolen card. Monopoly uses the resource hand
that is public under these two-player rules.

## Earlier failures retained

The first redesign, source `4f183c6`, dropped the useful nonwinning tactical
preference and simplified robber resource values. Against F it scored 105/200,
the same redesigned expert without learning scored 112/200, and original
OneVOne scored 125/200 on the matched boards. Directly against original it won
72/200. All ran without faults or overruns. Those initial results are retained
in `reports/learned_f_200.json`, `redesigned_heuristic_f_200.json`,
`original_f_200.json` and `learned_vs_original_200.json`.

The final source restores those priorities as an overridable prior, retains
resource-aware robber scoring, and retrains both models with new data. A
40-game development smoke scored 26/40 without faults; this was execution and
development evidence, not the final strength test.

One intermediate dataset used a CPU-timed tactical feature pass. A visibility
test exposed that scheduling could change which tactical features were found.
That dataset was discarded. The final feature pass uses a fixed node count,
and both networks were retrained on fresh regenerated data. Its unused manifest
is retained as `reports/discarded_timed_feature_data_manifest.json`.

The current runtime requires schema `onevone-visible-v2`. V1 checkpoints are
historical artifacts paired with source `4f183c6` and are rejected by the current
runtime. These results concern local F and the original OneVOne, not private O,
other teams or certified competition eligibility. This is supervised outcome
learning, with no online weight updates or opponent-history adaptation.

Reproduction commands and the custom-scorer interface are in [README.md](README.md).
