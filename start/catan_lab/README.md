# Catan agent lab

Five pinned upstream checkouts, two downloaded pretrained policies, two native
engines, and bounded CPU evaluation. **No model has been trained or fine-tuned
by this lab.** Catan remains an organizer hint; the actual game, allowed training
hardware, pretrained-weight policy and submission protocol are still unconfirmed.

Read [the findings and measured results](FINDINGS.md) first. The compact
[evidence snapshot](reports/pilot_summary.json) stays in Git. Full raw timing
samples, downloaded checkpoints, native builds and independent upstream Git
repositories are local ignored artifacts, restored with the commands below.

## Layout

```text
catan_lab/
  sources.json              exact upstream URLs and commits
  requirements.txt          additions to the ONE workspace environment
  setup_sources.py          small, immutable source checkouts
  fetch_checkpoints.py      selected weights, Git-blob and SHA256 verification
  setup_runtime.py          native builds outside upstream checkouts
  locks/                    resolved Cargo dependency lockfiles
  benchmark.py              isolated, time-bounded experiment runner
  worker.py                 repository-specific policy/environment adapters
  test_protocol.py          accounting and hidden-observation checks
  archive_report.py         compact evidence export
  reports/pilot_summary.json
  vendor/                   ignored independent upstream checkouts
  artifacts/checkpoints/    ignored pretrained weights + checksum manifest
  artifacts/native/         ignored native extensions and build metadata
  artifacts/reports/        ignored raw measurements
```

## Reproduce from `start/`

Use the existing `.venv`; do not create a Catan-specific environment. Requires
Git, a Rust toolchain, a C compiler, and internet for the first setup.

```bash
.venv/bin/python -m catan_lab.setup_sources
uv pip install --python .venv/bin/python -r catan_lab/requirements.txt
.venv/bin/python -m catan_lab.fetch_checkpoints
.venv/bin/python -m catan_lab.setup_runtime
.venv/bin/python -m unittest catan_lab.test_protocol -v
.venv/bin/python -m catan_lab.benchmark --games 32 --seconds 60
```

Source fetches skip historical model collections and run directories. Three
selected weight files total 25,553,900 bytes (24.37 MiB). Additional disk space
comes mainly from native compilation caches. Checkpoints are loaded on CPU with
PyTorch's `weights_only=True`; architecture/schema versions must match. Source
checkouts stay clean. Build copies include the macOS dynamic-symbol linking
flags; the original Eli binding lockfile needed resolution, so the resolved
lockfile is retained under `locks/` and subsequent builds use `--locked`.

The two Catanatron variants share a Python import name. Each runs in a separate
subprocess using its own source path; neither is installed over the other.
Native extensions are loaded from `artifacts/native/` in those workers.

## Useful experiments

```bash
# Pretrained MLP: 256 games per visibility mode, all four seats represented.
.venv/bin/python -m catan_lab.benchmark --agents eli --games 256 --seconds 60 --output catan_lab/artifacts/reports/eli_256.json

# Same engine/rules/opponents: compare cheap heuristic teachers.
.venv/bin/python -m catan_lab.benchmark --agents catanatron --classical-policy value --classical-opponents weighted --games 64 --output catan_lab/artifacts/reports/value_vs_weighted.json
.venv/bin/python -m catan_lab.benchmark --agents catanatron --classical-policy alphabeta --classical-opponents weighted --games 64 --output catan_lab/artifacts/reports/ab1_vs_weighted.json

# Transformer/GNN: untrained live-observation/action smoke, not a win-rate test.
.venv/bin/python -m catan_lab.benchmark --agents benjamin --games 2
```

`--threads` defaults to 1. `--seconds` bounds each worker's gameplay loop, with
a separate process timeout for imports/initialization. Capped games remain in
the all-attempts denominator. Model decision latency excludes environment
advancement; the report states its scope. Peak RSS includes Python/PyTorch, not
just weights. Different adapters handle forced actions differently, so their
latency medians are not controlled architectural comparisons.

The upstream Charlesworth fork imports its GUI from game logic and uses the
removed `np.int` alias. Its worker provides a headless GUI import shim and
restores that alias locally. Rules and policy weights are unmodified. Its
baseline uses three zero-logit conditional-head policies, which sample uniformly
within each head, not uniformly over all composite game actions.

## Next experiment

Once the organizer interface is known, choose one exact rule configuration,
freeze a seed/opponent/seat test suite, and collect teacher decisions and final
outcomes under that configuration. Compare a small Transformer against an MLP
on the same split and compute budget. Keep PPO and search improvements separate
from the initial imitation baseline. The current work is setup and evaluation.
