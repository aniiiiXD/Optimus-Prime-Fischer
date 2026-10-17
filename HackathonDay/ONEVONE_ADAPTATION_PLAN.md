# OneVOne → official game: baseline plan

## Objective

Measure the copied OneVOne heuristic against **F**, using the official engine,
observation masking, action-index interface and CPU clock. The first baseline
preserves the policy's scoring weights and tactical search limits.

## Steps

1. **Freeze a self-contained copy.** Put OneVOne, its three helper modules and
   its default configuration in `onevone_baseline/`. Record source hashes and
   the exact kit version. Replace imports that depend on the wider workspace.
2. **Match the game contract.** Two players, 10 VP, bank/port trades only. Read
   `obs`, obtain a permitted search world with `obs.sample_world(rng)`, and
   return the selected action's index in `playable_actions`. Keep hidden
   opponent development cards as samples, not facts.
3. **Use the official engine and harness.** Create a separate Python 3.11
   environment with the kit's exact engine, numpy, networkx and psutil pins.
   This heuristic imports no ML libraries, so unused torch/scipy/scikit-learn
   downloads are omitted for this local baseline. Pass
   `python -m catan_kit.selfcheck` before measuring the adapted agent. Install
   the full requirements if a later experiment uses those ML packages.
4. **Smoke test.** Run 20 games against `examples/f_agent.py`. Record all
   faults, timeouts and budget exhaustion; fix interface errors before the
   larger run.
5. **Measure the baseline.** Run 200 games on 100 fresh boards, both seats,
   against F. Save full evaluator output, exact command and seed range. Report
   win rate and its interval, CPU mean/worst, budget exhaustion and faults.
   This is an initial screen, not a final 1,000-game tournament estimate.
6. **Respond to measured failures.** If the clock is exhausted, measure the
   cost of tactical scanning and candidate evaluation before changing them.
   Compare any bounded-search variant with this frozen baseline on identical
   boards, then confirm the chosen variant on unused boards. Optimize policy
   strength only after the agent runs legally within the clock.

## What changes in the first copy

| Component | Adaptation |
|---|---|
| Entry point | `Agent()` and `decide(obs, playable_actions) -> int` |
| State access | Official sampled observation world |
| Dependencies | Local helper imports; official pinned engine |
| Config loading | Explicit local `config.json`; no implicit trained config |
| Rules metadata | Disable player trades and identify the official kit |
| Policy coefficients / search limits | Same values as the copied source |
| Training | None |

The official `obs.cpu_left` budget is deliberately not used to change policy
choices in this first measurement. Budget overruns remain visible in the
reported baseline rather than being hidden by a new fallback.

## Interpretation and eligibility

F is the official one-step value-function baseline. It is a different opponent
from a depth-two web bot. O is an organizer-trained bot; its implementation is
not supplied in this public kit, so these local runs cannot establish our
win rate against O.

This is a local diagnostic copy. The official rules require submitted work to
be built during the event, except attributed public material. Preserve the
copy's provenance; do not represent earlier private work as event-built work.
Confirm submission eligibility before uploading a derivative of that work.

Sources: [official logistics](LOGISTICS.md),
[official kit](https://github.com/lyric-hyde/catan-hackathon-agent/tree/04fffac90e00610f60e0b06d919f262f4b3ed9db).
