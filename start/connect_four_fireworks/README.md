# Fireworks GPU teacher audit

This is a deliberately small, **evaluation-first** experiment. A Fireworks-hosted
LLM sees legal Connect Four boards and returns a ranked list of columns. Our
local rules engine validates that answer, and the existing depth-5 alpha-beta
teacher supplies the comparison label.

The resulting JSONL is not automatically fed into Transformer training. An LLM
is a hypothesis generator, not a replacement for correct game rules or the
Pons solver. We first measure whether its top choices add useful diversity or
strategic signal over the search teacher.

## What each row contains

- the normal local alpha-beta policy/Q/value labels;
- the exact text prompt sent to Fireworks and its raw response;
- a parsed, legal LLM ranking when available;
- a soft policy derived from that ranking, kept separate from `policy_target`;
- request latency and token-usage metadata returned by Fireworks.

Malformed or illegal LLM answers are saved as invalid audit examples and do not
become targets.

## Run a bounded pilot

From `start/`, after placing `FIREWORKS_API_KEY=...` in `.env`:

```bash
set -a; source .env; set +a
.venv/bin/python -m connect_four_fireworks.generate_teacher_audit \
  --positions 12 --depth 5 --node-limit 5000 --max-tokens 512 \
  --output connect_four_fireworks/artifacts/data/fireworks_glm_flash_pilot.jsonl
```

The defaults use the accessible `accounts/fireworks/models/ember-1`
model, temperature zero, and a required typed tool call. `glm-5p3-flash` was deliberately
not selected as the default: in our initial 12-board trial it used the entire
180-token allowance for hidden reasoning and returned no answer content. The
command records the model
name, so later model comparisons remain fair.

The collector also retains boards with at least three legal columns by default,
so paid calls measure actual choice rather than forced endgame moves.

## Decision gate

Scale only if the pilot has near-perfect legal JSON compliance and its top-3
choices frequently contain the search teacher's best action. Even then, use
the LLM ranking as an additional input or candidate generator; preserve exact
search labels for the actual CPU Transformer training target.
