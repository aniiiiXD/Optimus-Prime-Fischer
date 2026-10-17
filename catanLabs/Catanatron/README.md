# Catanatron engine

The engine is already downloaded at
[`start/catan_lab/vendor/catanatron`](../../start/catan_lab/README.md). This
folder's [`engine.py`](engine.py) checks that checkout against its pinned commit
before a benchmark starts. We use the engine for rules, seeded boards and legal
moves; our future transformer supplies decisions.

To restore the ignored source on another machine, from `start/` run:

```bash
.venv/bin/python -m catan_lab.setup_sources
```

The evaluation preset is four-player base Catan, 10 victory points, standard
seven-card discard limit, no friendly robber, and official spiral number
placement. The source is GPL-3.0; see the upstream
[repository](https://github.com/bcollazo/catanatron).
