# Catan observation and evaluation map

The repository contains two different two-player Catan paths. Confirm which one the user means before reporting results.

| Path | Source of truth | What it is |
|---|---|---|
| `HackathonDay/official-kit/` | `catan_kit/obs.py`, `match.py`, `evaluate.py`, `examples/f_agent.py` | Downloaded official kit, ignored by Git. Its pinned Catanatron dependency supplies the engine. |
| `HackathonDay/onevone_baseline/` | `agent.py`, `onevone.py`, `tactical_vp.py`, `fair_value.py`, `config.json` | Self-contained official-interface diagnostic policy. |
| `HackathonDay/reports/` | Run manifests, outputs, and decision traces | Local measured evidence; inspect seeds, hashes, faults, and CPU before quoting a score. |
| `catanLabs/OneVOne/` | `agent.py`, `evaluate.py`, `README.md`, `experiment_catalog/README.md` | Earlier local Catanatron experiments with a separate evaluator and rules assumptions. |
| `catanLabs/Transformer/`, `catanLabs/External/` | Each folder's README, training and evaluation scripts | ML experiments. Check whether a checkpoint is loaded by the actual playing entry point. |

## Official observation contract

`Agent.decide(obs, playable_actions) -> int` returns an index into legal actions. `obs` exposes `board`, `turn`, `me`, `cpu_left`, `bank`, `my_resources`, `my_devs`, `my_public_vps`, `my_actual_vps`, `opp_resources`, `opp_dev_count`, `opp_public_vps`, `played_devs`, and `sample_world(rng)`. The opponent's unplayed development-card identities, hidden VP, and future deck order are not observed. `sample_world` supplies one consistent hypothetical game for search.

The supplied local opponents are `examples/f_agent.py` and `examples/random_agent.py`. The organizers' O bot is private; do not imply a local test covers O.

## Useful commands from the Hyde root

Run the pinned baseline with a new label and unused seed range; this script records a manifest and refuses to overwrite a result:

```sh
HackathonDay/.venv/bin/python HackathonDay/evaluate_baseline.py --games 20 --seed0 2026300000 --label observation_smoke
```

To test a different official-interface agent against F, run from `HackathonDay/official-kit/`:

```sh
../.venv/bin/python -m catan_kit.evaluate --a /absolute/path/to/agent.py --b examples/f_agent.py --games 1000 --seed0 2026301000
```

For a direct match against OneVOne, change `--b` to `../onevone_baseline/agent.py`. Each even game count plays every board from both seats. Read `HackathonDay/LOGISTICS.md` for the official runtime, setup, and submission constraints. The ignored official kit and local environment may be absent in a fresh clone.
