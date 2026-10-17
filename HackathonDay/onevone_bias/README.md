# OneVOne: official-harness diagnostic copy

`agent.py` is the official entry point. The other four Python files contain
the copied heuristic and its helper classes. `config.json` holds the frozen
parameters. No training is performed.

See [the adaptation plan](../ONEVONE_ADAPTATION_PLAN.md) and
`PROVENANCE.json`. This directory imports no modules from `catanLabs`.

From the official kit root, with the separate environment active:

```sh
python -m catan_kit.evaluate --a ../onevone_baseline/agent.py --b examples/f_agent.py --games 20 --seed0 2026101700
python -m catan_kit.evaluate --a ../onevone_baseline/agent.py --b examples/f_agent.py --games 200 --seed0 2026101800
```

Both commands enforce the official CPU clock. The first covers 10 paired
boards; the second covers 100 different paired boards. Results belong to
these rules and F opponent, with CPU measured on the local development machine.
