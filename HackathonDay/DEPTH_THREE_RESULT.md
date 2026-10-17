# Depth-three OneVOne experiment

Branch: `experiment/depth-three`. The frozen `onevone_baseline/` on `main` was
not changed. `depth_three_candidate/` adds a bounded search over at most three
of our own deterministic actions after rolling. It can find sequences such as
trade → trade → city or Road Building → road → road. It does not simulate an
opponent reply, a future dice roll, a hidden development draw, or robber theft.

The planner runs after the existing FairValue and two-action TacticalVP choice.
It returns a new first action only when a searched sequence gives more *known*
victory points than that immediate choice. The final implementation searches
once per turn, up to 240 action expansions and 24 ms of agent CPU when its
remaining allowance permits. Its score is still heuristic; no ML model is
loaded by this official-interface candidate.

## Official-harness tests

All rows use the pinned official kit `04fffac90e00610f60e0b06d919f262f4b3ed9db`,
the local official engine, both seats per board, and the enforced CPU clock.
No test had an illegal action, exception, timeout, or crash.

| Build | Opponent / seeds | Candidate wins | Matched OneVOne wins | Candidate clock overruns | Mean candidate CPU/game |
|---|---|---:|---:|---:|---:|
| Initial depth-three search | F / 2026301000–2026301099, 200 games | 120 | 115 | 0 | 279 ms |
| Search reordered for three-action wins | F / 2026303000–2026303099, 200 games | 126 | 125 | 1 (4 random fallback moves) | 266 ms |
| Once-per-turn, higher CPU reserve | F / same 2026303000–2026303099, 200 games | 124 | 125 | 2 (17 random fallback moves) | 387 ms |
| Initial search, direct head-to-head | Frozen OneVOne / fresh 2026302000–2026302099, 200 games | 99 | 101 | 0 | 207 ms |

The initial candidate gained 10 and lost 5 individual games versus control on
the first F schedule. The final candidate gained 6 and lost 7 on the second.
Those differences are small and do not establish stronger play. The direct
head-to-head result is effectively even. The later candidate's CPU failures
are a separate reason not to promote it. CPU means vary across sequential
local runs, so the clock failures are more informative than a precise mean
comparison.

As a diagnostic, the reordered planner found an alternative on all **13/13**
previously saved positions with a known winning line, including four lines of
three actions. Those positions came from the earlier OneVOne trace and were
used to adjust search order; they are not fresh win-rate evidence.

The official F comparison is local. The organizers' O bot is unpublished and
was not tested. The copied pre-event OneVOne policy and its derivatives remain
diagnostic under the event's build-during-event rule.

## Decision and ML status

Keep the candidate on this experiment branch. The best clean F screen was
120/200 versus 115/200, but a fresh direct match was 99/200. The final
version was 124/200 versus 125/200 with two clock overruns. No depth-three
variant qualifies to replace the frozen OneVOne agent.

The older local MLP is a separate, unpromoted research model: its documented
screen won 3/64 against the OneVOne heuristic. This branch did not train or
run an ML model. A useful next ML test would learn a cheap value or branch
ranking from official-observation game traces and then compare completed-game
wins and clock use on fresh paired boards; model accuracy alone is not a
promotion criterion.

## Evidence and reproduction

Full game records and source hashes for the final version are in
`reports/depth_three/clock_safe_f_200.json`. Earlier exploratory records are
in the same folder. Their evaluator recorded entry-point hashes but did not
hash helper modules, so the reports distinguish the builds and seed schedules
but do not fully reconstruct every intermediate source revision.

The official kit and virtual environment are locally ignored. From this
worktree, link or restore them under `HackathonDay/` as documented in
`LOGISTICS.md`, then run:

```sh
HackathonDay/.venv/bin/python HackathonDay/evaluate_depth.py \
  --a HackathonDay/depth_three_candidate/agent.py \
  --b HackathonDay/official-kit/examples/f_agent.py \
  --games 200 --seed0 2026306000 \
  --output HackathonDay/reports/depth_three/repeat_f_200.json
```

Choose a new output name and seed range for every run. A direct match uses
`--b HackathonDay/onevone_baseline/agent.py`.
