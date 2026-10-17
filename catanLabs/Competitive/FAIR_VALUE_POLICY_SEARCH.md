# Fixed FairValue policy search (deferred negative control)

The runner `tune_fair_value.py` tests six fixed perturbations of four strategic
weights: immediate reachability, buildable sites, hand synergy, and army size.
Every multiplier is 0.5 or 1.5 relative to `fair_army_finish.json`; the
incumbent file stays untouched. Each variant and the Fair control has a config
SHA-256 in the run manifest. This is a small direct-outcome search, not a
trained model or a promotion test.

## Fixed design

| Stage | Board seeds | Arms | Matches |
| --- | --- | --- | ---: |
| Wiring smoke | 20275951 | Fair + six variants | 56 |
| Selection stage 1 | 20276000–20276003 | Fair + six variants | 224 |
| Selection stage 2 | 20276100–20276107 | Fair + stage 1 top two | 192 |

Each board rotates the candidate through four seats in both existing opponent
mixes (`mixed_diverse`, `mixed_strong`), under the pinned four-player,
first-to-10 Catanatron v5 protocol. Every agent uses the same board and
opponent assignment in a matched row. Stage 2 is independent of stage 1; its
results are **not pooled** with stage 1 for a confidence interval. The runner
selects by equal-mix paired win-rate difference from Fair, then equal-mix
paired final-VP difference, then variant name. No stage constitutes a claim of
improvement. Complete games with zero illegal actions/errors/caps are required
before a selection file is written.

Per-game atomic partial JSON files survive interruption. The runner checks a
20-second game cap, 500 turns, and 15,000 actions, and aborts on any cap. It
also stops at a 540-second aggregate wall or process CPU budget with a
five-second shutdown margin. The two diagnostic smoke revisions consumed
18.343 + 18.958 = **37.301 wall seconds** and 18.095 + 18.905 = **37.000 CPU
seconds**. A 540-second full run would therefore remain below 590 aggregate
seconds for the experiment. `--output-dir` must be new and empty so stale
selection files cannot be mistaken for fresh results.

Exact full command, **deferred and not run**:

```sh
PYTHONHASHSEED=0 start/.venv/bin/python catanLabs/Competitive/tune_fair_value.py \
  --output-dir catanLabs/Competitive/runs/fair_value_policy_search_full \
  --wall-budget 540 --max-game-seconds 20
```

At most 416 complete tuning games are expected. If the budget expires, the
runner leaves partials and an `incomplete_no_selection` status; it does not
write `selection.json`. Before changing the incumbent, freeze the selected
config hash and run a **fresh, at least 100-board** matched confirmation against
Fair (including direct Fair opponent games and uniform strong-opponent games).
Use new seeds, all four seats, and report paired uncertainty and decision
latency. Do not choose a new config after looking at those confirmation games.

## Wiring evidence and limitations

The first 56-game smoke on seed 20275950 tried production and opponent
production ±25%; all four such variants matched Fair's actions in all eight
paired games each. Based on those action traces, before any selection games,
the fixed design moved to the four lower-scale weights above. The second
56-game smoke on seed 20275951 completed all games, with no errors/caps, in
18.958 wall / 18.905 CPU seconds. The two reach-increasing variants first
diverged from Fair in 1 of 8 paired games each; the other four did not diverge
on that board. This measures **wiring and local policy sensitivity only**.
The single-board win counts have no model-selection value. A subsequent
same-state sensitivity probe played eight complete Fair-driven games on two
unused `mixed_strong` boards (20275952–20275953, all four seats). Across 706
Fair decisions, changing the late development-card purchase bonus by ×0.5 or
×1.5 produced **zero** action flips. Changing next-step reachability by ×1.5
produced **one** early-game flip; ×0.5 produced zero. Reach-plus-army ×1.5
also produced zero on those boards. The probe took 6.171 wall seconds. Its
machine-readable report is `runs/fair_value_policy_sensitivity_probe.json`.

**Decision:** defer the 416-game tuning search. The diagnostic evidence says
this ±50% neighborhood is mostly flat under FairValue's action scores, so
additional full-game comparisons would likely measure identical policies.
Retain the runner as a reproducible negative-control experiment. Total
diagnostic compute used was **43.472 wall seconds** (37.301 for two smokes +
6.171 for sensitivity); no stage-one or stage-two tuning games were run.
