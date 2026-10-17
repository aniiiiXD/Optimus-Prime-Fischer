# Hackathon Day

The [depth-three OneVOne experiment](DEPTH_THREE_RESULT.md) lives on this
branch. Its tested candidate was not promoted to the frozen baseline.

This is a fresh log recording work done after the reset. Documentation entries
are labeled separately from experimental results. Earlier experiments remain in their original folders;
they are not part of this report.

See [LOGISTICS.md](LOGISTICS.md) for the official setup, interface, runtime
limits, evaluation, deadlines and submission checklist.

The [OneVOne adaptation plan](ONEVONE_ADAPTATION_PLAN.md) describes the copied
agent, interface changes and official-harness baseline protocol.

The [first baseline result](ONEVONE_BASELINE_RESULT.md) records 127/200 wins
against official F, followed by 609/1,000 on fresh paired boards. Both runs
had zero faults or budget overruns on the local machine.

The [win/loss analysis](ONEVONE_WIN_LOSS_ANALYSIS.md) compares all 200 games,
four component-removal experiments, and concrete late-game decision mistakes.
The baseline is unchanged; V2 improvements have not been implemented.

Open `index.html` to read the report. Edit `report.json` to update it, then run:

```sh
python3 HackathonDay/render.py
python3 HackathonDay/render.py --check
```

## Play 1,000 local games

From the Hyde repository root, run the pinned official evaluator against the
supplied F bot:

```sh
HackathonDay/.venv/bin/python HackathonDay/evaluate_baseline.py --games 1000 --seed0 2026203000 --label baseline_f_1000_repeat
```

The command tests `onevone_baseline/agent.py`, plays 500 board seeds from both
seats, enforces the official CPU clock, and saves the full output and run
metadata under `reports/onevone_baseline/`. Choose a new label and seed range
for another run. The recording script refuses to overwrite an earlier result.

To test a different strategy, point the official evaluator directly at its
`agent.py` and choose the opponent with `--b`:

```sh
cd HackathonDay/official-kit
../.venv/bin/python -m catan_kit.evaluate --a /absolute/path/to/your/agent.py --b examples/f_agent.py --games 1000 --seed0 2026201000
```

For a direct candidate-versus-OneVOne match, use:

```sh
../.venv/bin/python -m catan_kit.evaluate --a /absolute/path/to/your/agent.py --b ../onevone_baseline/agent.py --games 1000 --seed0 2026202000
```

F and OneVOne are playable local opponents; the organizers' O bot is not
distributed with the kit. Results against O require the event's private
evaluation service. Every candidate, including an ML policy, needs an
`Agent.decide(obs, playable_actions)` adapter to play in this harness.

Add one object to `entries` for each new attempt, including failures:

```json
{
  "id": "unique-short-id",
  "date": "2026-10-17",
  "title": "What we tried",
  "status": "planned",
  "objective": "What question this experiment answers",
  "setup": "Rules, opponent, seeds, seats, agent version and comparison",
  "result": "Measured wins/games and any errors; use Pending before a run",
  "learning": "What the result does and does not establish",
  "next_step": "What we will do next",
  "evidence": "relative/path/to/log-or-report"
}
```

Statuses: `documented`, `measured`, `planned`, `running`, `improved`, `no-gain`, `failed`, or
`inconclusive`. `evidence` may be omitted until a log exists. Record results
only after running the experiment; label predictions as predictions.
