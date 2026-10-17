---
name: observe-agents
description: >-
  Inspect agents and game frameworks in the Hyde repository: show what an agent
  observes, trace its decision path, identify active models and engines, and
  explain measured match behavior. Use for understanding or comparing existing
  agents, not for building a new policy.
---

# Observe agents and frameworks

Make the running system inspectable from source code and recorded games. Separate verified behavior from design notes and proposed behavior.

## Locate the exact system

1. Identify the requested game, agent, branch or worktree, and ruleset. Check `git status --short --branch` before interpreting files. Start with `README.md`, `start/README.md`, `catanLabs/README.md`, or `HackathonDay/README.md` as appropriate.
2. Find the actual entry point, configuration, model or weights, engine, observation encoder, action adapter, and evaluator. Use `rg --files` and `rg -n` to locate them. Follow imports and configuration loading to learn which component runs by default. An available model or checkpoint is not evidence that the playing agent uses it.
3. For the official two-player Catan kit versus older local Catan experiments, read [references/catan.md](references/catan.md). Do not pool their win rates or assume they share an observation or action interface.

## Show one decision

- Start at the function called by the game harness. Show its inputs, the legal-action list, the selected action or index, and the path through any heuristic, search, or neural model.
- Point to the exact source defining visible fields and hidden fields. If a world is sampled for search, label its hidden values as guesses, not observations.
- When a concrete decision is requested, use an existing trace or replay tool if present. Report the board seed, turn, seat, observation, legal choices, selected choice, and relevant score or model output. Say when a requested intermediate value is not logged rather than inventing it.

## Observe performance

- Prefer the project's existing evaluator and saved reports. If a new run is requested, start with a small legal/clock smoke test and then use the requested game count. Preserve the agent and opponent versions, rules, seed range, both seats, errors, timeouts, caps, and runtime in the result.
- For comparisons, use the same paired boards and opponent for candidate and control; confirm promising changes on unused seeds. Keep local results separate from private leaderboard scores and organizer hardware timings.
- Inspect training data, checkpoints, and train/eval split only when the question concerns ML. Distinguish training accuracy from completed-game win rate.

## Respond with evidence

Give a compact map of **observation → decision → action → engine → evaluator**, with clickable source and report links. State which components are active, the last measured result's opponent and denominator, and what has not been measured. Do not claim a bot has learned opponent bias without a stored update mechanism or a measured adaptation test.
