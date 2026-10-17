# Catan labs

Our first Catan Transformer has been trained from a fresh initialization. We
then built a stronger CPU strategy using fair chance handling, resource and
expansion parameters, and development-card timing. This folder keeps the
engine, models, strategy agents, artifacts, and evaluation protocol inspectable.

For all two-player bot variants, measured experiments, and failure analyses,
start with the [1v1 experiment catalog](OneVOne/experiment_catalog/README.md).
It also maps the variants kept in the separate
`Hyde-catan-experiments` worktree.

Start with the [visual progress report](index.html) to see what is in place,
what the early results do and do not prove, and the proposed learning loop.

For the next implementation session, use the [agent execution pipeline](NEXT_AGENT_PIPELINE.md).
It records the required competition facts, dependency gates, bounded runs,
promotion criteria, and the first concrete ticket.

The [last 24 hours handoff](LAST_24_HOURS_HANDOFF_2026-10-16.md) gives the
current incumbent, exact match results, completed CPU experiments, known
limits, and an ordered starting point for another agent.

The [hackathon deck strategy](HACKATHON_DECK_STRATEGY.md) turns the newly
shared Catan rules slides into concrete playing ideas, ranked experiments,
and an event-day contract check.

- [Catanatron](Catanatron/README.md): pinned game engine and legal-action source.
- [Benchmarks & Evals](Benchmarks&Evals/README.md): repeatable four-player tournament.
- [Transformer](Transformer/README.md): first from-scratch CPU behavior-cloning run and saved checkpoint.
- [Competitive CPU agent](Competitive/README.md): current strategy, parameter experiments, and held-out results.
- [Policy-guided search experiment](Competitive/SEARCH_EXPERIMENT.md): two-action search, Transformer prior ablation, and matched confirmation tournaments.
- [External CPU PPO + rollout reproduction](External/README.md): the published 82% project built on this Mac, with seat-rotated first-to-7 and first-to-10 tests, a CPU fine-tune, and explicit rules/visibility limits.
- [PPO policy transfer to Catanatron](External/TRANSFER_EXPERIMENT.md): explicit observation/action bridge, visibility checks, and a matched 10-point tournament showing the port trails the fair-value incumbent.
- [ML research and plan](ML_RESEARCH_AND_PLAN.md): data coverage, model comparisons, CPU training, PPO and promotion criteria.

The earlier [`start/catan_lab`](../start/catan_lab/README.md) is a source audit
and upstream-agent pilot. Its downloaded Catanatron checkout is reused here.
The external PPO weights are evaluated through a separate transfer adapter;
they are not the original from-scratch Catanatron Transformer checkpoint.
