# Catan From Scratch

A clean workspace for building and evaluating a competitive 1v1 Catan agent.
The primary measure is match wins under the actual game rules and action API.

## Rules source

The starting rules reference is the [Hyde@IITB introductory Catan deck](../../../../Downloads/Hyde@IITB.pdf)
(11 slides, about 61 MB). It describes a two-player game with five resources,
dice-driven production, roads, settlements, cities, development cards, the
robber, and bank/port trading. The first player to 10 victory points wins.

The deck is an overview, so it does not define the competition's observation
format, legal-action API, compute limits, or every edge case. Treat the details
in `DESIGN.md` as the initial rules baseline and confirm the full contract
before building the simulator adapter or final dataset.

## Competition brief

The WhatsApp announcement says the agent will face organizer reference bots
and other teams. Dice, hidden information, trades, and limited compute are
called out explicitly. Search, self-play, learned value functions, imitation,
and handcrafted heuristics are all allowed. Judges care about consistent wins
and a clear explanation of the strategy. The full problem statement, constraints,
submission rules, and evaluation details are due at the event.

## Workspace

- `DESIGN.md` — objectives, evaluation rules, and build sequence.
- `src/` — agent, simulator adapter, and training code.
- `configs/` — rules, model, and experiment settings.
- `data/raw/` — generated trajectories and their manifests.
- `data/processed/` — versioned training and validation datasets.
- `runs/` — checkpoints and training logs.
- `reports/` — match results and analysis.
- `tests/` — rules, feature, and agent checks.

This is the starting scaffold. No dataset, model, or policy code has been
copied into it. The current `OneVOne` project remains available as a benchmark
reference while this design is worked through.
