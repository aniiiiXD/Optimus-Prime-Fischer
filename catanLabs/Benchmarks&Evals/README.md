# Benchmarks & Evals

This is the fixed exam for our future Catan transformer. It does **not** train
a model. Every opponent tier uses fresh seeded boards, with our candidate
rotated through all four seats on each seed. A game that errors or hits a cap
stays in the attempted-games denominator. Reports include rules, engine commit,
agent code/checkpoint hashes, each game result, win rates by opponent and seat,
and decision latency.

The current four-tier protocol is **v5**. It isolates the RNG in hypothetical
`Game.copy()` states so search cannot advance the live game's dice stream. It
passes explicit dice/card outcomes correctly in Catanatron's search branches;
upstream `Game.execute()` ignores their `Action.value` without an action record.
It also fixes `PYTHONHASHSEED=0` at process start and sorts legal actions by a
canonical key to keep ties repeatable across runs. Older v1–v4 JSON reports
remain historical experiments and should not be mixed with v5 results. The
pinned engine source itself is unchanged.

From the workspace root, run a short smoke tournament:

```bash
start/.venv/bin/python 'catanLabs/Benchmarks&Evals/run.py' \
  --agent random --opponents random --boards 1
```

After we have an agent, run the frozen opponent ladder:

```bash
start/.venv/bin/python 'catanLabs/Benchmarks&Evals/run.py' \
  --agent-file path/to/our_agent.py --checkpoint path/to/model.pt \
  --opponents random,weighted,value --boards 100 \
  --report 'catanLabs/Benchmarks&Evals/reports/transformer_final.json'
```

The agent file must define `create_agent(color, checkpoint, seed)`. It returns an
object with `decide(game, playable_actions)` that chooses one of the supplied
legal actions. The candidate is constructed separately for each game. Its
decision code must use only information visible to that player; Catanatron's
low-level `game` object contains hidden state, and this harness cannot enforce
that boundary yet. The benchmark rejects illegal returned actions. See
[`agent_template.py`](agent_template.py) for the interface. Player-to-player
trade offers are outside this first protocol because they are not in the
ordinary legal-action list; bank and port trades remain available.

The `random` tier is a basic functionality check, `weighted` is a cheap
strategy baseline, and `value` is a stronger hand-built baseline. An optional
`alphabeta1` tier uses Catanatron's depth-one search bot and costs more time:

```bash
start/.venv/bin/python 'catanLabs/Benchmarks&Evals/run.py' \
  --agent-file path/to/our_agent.py --checkpoint path/to/model.pt \
  --opponents alphabeta1 --boards 100
```

These are
comparisons within this one ruleset, not a public leaderboard or a claim of
human-level strength. Keep training boards separate from the evaluation seeds.
