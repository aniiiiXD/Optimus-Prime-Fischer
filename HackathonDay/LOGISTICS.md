# Hackathon logistics

Verified on 17 October 2026 against [the official kit](https://github.com/lyric-hyde/catan-hackathon-agent/tree/04fffac90e00610f60e0b06d919f262f4b3ed9db), commit `04fffac90e00610f60e0b06d919f262f4b3ed9db`. This is an operating checklist, not an experiment result. No agent games were run to prepare it.

## Event, selection and deliverables

| Item | Published requirement |
|---|---|
| Date / team | 17 October 2026; 3–4 people |
| Game | Two players; 10 VP; randomized official spiral board; bank/port trades only |
| Session | 10:00–19:30 IST |
| Agent uploads | 11:00–17:00; 15:30 checkpoint eliminates nobody |
| Report | Top 30 teams: PDF by 17:30; ≤4 A4 pages excluding references, single column, ≥11 pt |
| Results / presentations | Results 17:50; presentations start 18:00; 6 minutes + 4 minutes questions |
| Limits | 7 scored submissions, 10 total uploads including failures; failed preflight consumes no scored slot |
| Selection | Best submission by private-set mean win rate against F and O; top 30 advance |
| Groups | Three groups of ten; 1,000 games per opponent pairing; top two per group present |
| Robustness | Mean win rate against your five weakest opponent matchups |
| Strength | `(win_rate_vs_F + win_rate_vs_O) / 2` |

Public leaderboard position does not determine selection. F is the supplied value-function bot; O is the organizers' trained bot. Robustness/strength and report/presentation carry roughly equal combined weight.

Submit work built during the event; public libraries, papers and open-source code are exceptions requiring attribution. No sharing between teams, runtime LLM/service calls, hidden-information access or harness interference. Report only measurements from actual runs.

Source: [problem statement, §§4, 8–11](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/PROBLEM_STATEMENT.md).

## Access and environment

The provided development machine has **8 CPU cores and 16 GB RAM**. Log in using `team<teamname>`; obtain credentials from the registration instructions. Change the initial password with `passwd`. Do not store credentials in this repo.

```sh
ssh team<teamname>@32.190.233.160
```

Expected gateway ED25519 fingerprint:

```text
SHA256:Z8tsuVPiJ3dyI2vOQt2wV8hcBQxyaZOLpB2H4rnvvBE
```

The kit and environment are already installed remotely. Run these commands individually:

```sh
cd ~/catan-agent-hackathon
source .venv/bin/activate
python -m catan_kit.selfcheck
```

The [team portal](https://main.d2sgimrfzis7m5.amplifyapp.com/) uses the normalized team name without `team` and the leader's registered phone number. It shows submission status, scores and leaderboard. Gateway authentication uses passwords; adding an SSH key does not replace it. OpenCode setup is optional; keep API keys outside code and submissions.

Source: [README: machine, portal and assistant setup](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/README.md).

### Local installation, when needed

Use Python **3.11** and the official requirements. The README's clone command names a different repository; this checklist uses the repository supplied to us and inspected above.

```sh
git clone https://github.com/lyric-hyde/catan-hackathon-agent.git
cd catan-hackathon-agent
python3.11 -m venv .venv
source .venv/bin/activate
```

On Linux/Windows, install CPU PyTorch before the requirements; macOS skips this separate command:

```sh
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

```sh
pip install -r requirements.txt
python -m catan_kit.selfcheck
```

Pinned engine commit: `d3f4ad05bb78d8b2309631d6d3cfa8fcb6fda816`. Do not replace it with a PyPI release: action formats differ. Package pins: numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, torch 2.8.0, networkx 3.6.1 and psutil 7.2.2. Agent dependencies must follow the permitted package list in the problem statement; a harness dependency is not blanket permission to use it in an agent.

Sources: [requirements](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/requirements.txt), [installation instructions](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/README.md).

## Interface and observations

Submit `agent.py` exposing:

```python
class Agent:
    def __init__(self):
        pass  # Load weights or tables once.

    def decide(self, obs, playable_actions) -> int:
        return 0  # Replace with our policy; return a legal list index.
```

This is the required interface, not a competitive policy. Source: [starter agent interface](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/README.md).

| Information | Access |
|---|---|
| Board, ports, pieces and robber | `obs.board` |
| Own resources / development cards | `obs.my_resources`, `obs.my_devs` |
| Own public / actual VP | `obs.my_public_vps`, `obs.my_actual_vps` |
| Opponent resources / public VP | `obs.opp_resources`, `obs.opp_public_vps` |
| Opponent unplayed development count | `obs.opp_dev_count` |
| Bank / played development cards | `obs.bank`, `obs.played_devs` |
| Turn / color / remaining CPU seconds | `obs.turn`, `obs.me`, `obs.cpu_left` |
| Search world | `obs.sample_world(rng)` |

Opponent unplayed development identities, hidden VP and deck order remain unknown. Search worlds randomize those unknowns consistently with visible information. Opponent resources are exactly derivable from bank and own hand. Never treat sampled hidden cards as observed truth.

Source: [observation implementation](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/catan_kit/obs.py).

## Runtime budget

| Constraint | Limit / behavior |
|---|---|
| Per-agent CPU | `min(0.300 + 0.0125 × decisions, 1.500)` seconds per game |
| Budget exhausted | Random legal moves finish that game |
| Per-decision response | 3 seconds maximum |
| Initialization | 60 seconds; outside game CPU clock |
| Evaluation memory | 2 GB per agent |
| CPU allocation | One core; more threads do not add capacity |
| Forced action | Harness executes it without calling our agent |
| Illegal index / exception | Recorded fault and random legal move |

Use a real integer index. Budget search using `obs.cpu_left`; copying, receiving observations and evaluating successors consume agent time. Initialization happens per process, not per game. A process can persist across games, so reset game-specific state deliberately.

Sources: [match harness](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/catan_kit/match.py), [agent host](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/catan_kit/host.py).

Runtime is Python only: no bundled executables/extensions or child programs. No network; read submission files and installed packages, write only to `tempfile.gettempdir()`. More than 2% illegal-index/exception decisions disqualifies a submission. Official hardware timings determine compliance. Source: [submission/runtime rules](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/PROBLEM_STATEMENT.md).

## Testing and recording

Run from the official kit root. Use even game counts for complete seat pairs:

```sh
python -m catan_kit.evaluate --a my_team/agent.py --b examples/f_agent.py --games 20
python -m catan_kit.evaluate --a my_team/agent.py --b examples/f_agent.py --games 1000 --seed0 1200000
```

**1,000 games means 500 boards played twice**, with reversed seats and matching dice. Defaults are 10 VP and seed start 990000. F is the supplied `examples/f_agent.py` baseline. Fresh confirmation needs a different seed range from tuning. The harness reports wins, interval, CPU, budget overruns, errors, timeouts, crashes and load time. A game reaching 1,000 turns without a winner counts as a loss for each agent.

Source: [evaluation script](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/catan_kit/evaluate.py), [turn cap](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/catan_kit/match.py).

Our workflow: save the agent/configuration, command, kit commit, seed range and full output for each run; record successes, failures and uncertainty in `report.json`. Before promotion, compare candidate and control under the same harness and seed schedule, then confirm on unused boards. These are our proposed working practices.

## Packaging and uploading

Put `agent.py` at the ZIP root beside required weights and helper files; **maximum 100 MB**. Resolve assets relative to `__file__`; Linux filenames are case sensitive. Verify the archive before upload.

```sh
cd my_team
zip -r ../my_team.zip agent.py
```

The command above packages only `agent.py`; add each required asset explicitly. On the team machine:

```sh
~/upload.sh my_team.zip
~/upload.sh report.pdf
```

Check the upload confirmation and portal status. The latest PDF is the report. Keep a local upload ledger because total attempts are limited.

Source: [README: submission procedure](https://github.com/lyric-hyde/catan-hackathon-agent/blob/04fffac90e00610f60e0b06d919f262f4b3ed9db/README.md).

## Next actions

1. Access the team machine and pass the official self-check.
2. Create our event agent against this observation/index interface.
3. Run the 20-game smoke test; fix faults and clock overruns.
4. Evaluate against F on fresh paired boards; log every attempt.
5. Package, verify and submit within the published deadline; prepare the evidence-backed PDF in parallel.
