# Learning Catan from scratch on CPU

Research and experiment plan, 15 October 2026

## Current measured status

The initial 10-minute CPU Transformer pilot was legal but weak. We then tuned
a fair CPU value/search policy and corrected the benchmark's chance handling.
Under corrected protocol v5 on 24 held-out board seeds with all seats rotated, that
policy won 32/96 against three Value bots and 28/96 against three AlphaBeta1
bots. It also won 38/96 in a mixed Weighted + Value + AlphaBeta1 table and
31/96 against Value + two AlphaBeta1 bots. The [full strategy report](Competitive/RESULTS.md)
has the same-board Value controls, uncertainty estimates, and code. This is
the current playing-strength target for later learned policies. Historical
Transformer and teacher pilot scores used v1 and should not be numerically
merged with v5 results.

A later [policy-guided search experiment](Competitive/SEARCH_EXPERIMENT.md)
tested a second action after promising roads, trades, and resource-granting
cards. Across 48 new boards it won 88/192 against three AlphaBeta1 bots,
versus 64/192 for the fair policy on matched boards. It was roughly even
against Value when uncertainty is included and did not improve the stronger
mixed-opponent table. The pilot Transformer used as a branch prior gave no
clear gain over the fair-policy search. No new neural training was done.

We then built and ran the [published CPU PPO + rollout system](External/README.md)
in its own fast Rust engine. Its shipped MLP plus full-game search won 145/192
in first-to-7 games against three upstream heuristics with all seats rotated.
We resumed that network for about 501 CPU seconds total, adding 9.51 million
decisions in the corrected run. The standalone 10-point policy improved by
18.8–22.7 win-rate points against two upstream heuristic families on matched
seeds. The original trade-filtered search did not improve, but letting search
consider trade offers won 164/192 against evolved upstream bots versus
146/192 for the shipped search on two board sets. This is strong evidence that
CPU training and search can work in a fast Catan engine. It is **not** a
Catanatron result: the upstream search sees hidden information and uses a
different action/rules interface. The next transfer problem is a visible-only
policy and search loop under our actual competition rules.

## The main idea

We can train our own model weights from random initialization on CPU. “From
scratch” does not have to mean making the model discover every rule through
random play. A practical learning path is:

```text
our Catan engine
      ↓
mixed bots play games and produce varied states
      ↓
teacher choices + complete-game outcomes
      ↓
small MLP and small Transformer learn from the same data
      ↓
PPO/self-play tries to improve the best supervised model
      ↓
the frozen tournament checks whether play actually improved
```

This follows the useful part of the Connect Four work: collect varied legal
states, remove true duplicates, mark tactical/strategic patterns, cache labels,
keep whole games out of the test split, and compare every new model on the same
held-out arena. The important Catan change is that “position variety” must cover
game phase, board geometry, player resources, public information and strategic
situations—not just different board layouts.

## What prior work teaches us

| Work | Useful lesson | Limit when applying it here |
|---|---|---|
| [Strategic Dialogue Management with Deep RL](https://arxiv.org/abs/1511.08099) | A small DQN-style MLP learned four-player trade offers and replies; it won 53.36% against three heuristic traders over 10,000 test games, using about 500,000 training experiences. | It learned only trade dialogue; JSettlers controlled the rest of play. Its offer space and opponents differ from our current harness. |
| [Cross-dimensional Catan network](https://arxiv.org/abs/2008.07079) | A neural network can combine board, player and action information; its paper reports beating JSettlers. | The study uses a simplified two-player game without player trading. It is not a standard four-player benchmark. |
| [Deep Catan](https://rlg.mlanctot.info/2022/papers/AAAI22-RLG_paper_31.pdf) | CNN learning plus Expert Iteration/search is a useful policy-improvement recipe. | Its comparison and rule setup are specific to that study. |
| [Charlesworth’s Catan RL project](https://github.com/henrycharlesworth/settlers_of_catan_RL) | Four-player PPO, structured action heads, attention, and a candid record of what training did and did not learn. | Its author reports that the agent was not superhuman and struggled with trading. |
| [BenjaminL1’s current Catan RL code](https://github.com/BenjaminL1/catan_rl) | A tile Transformer plus graph encoder and factored action heads are a concrete board-aware design to study. | It targets a Colonist-style 1v1 variant; the pinned checkpoint we inspected has no trained weights. |
| [Eli6th’s Catan RL project](https://github.com/Eli6th/catan-rl) | Fast simulation and policy-guided search can matter as much as a larger network. | Its headline uses a 7-point game and perfect-information search, so it is not directly comparable to our hidden-information test. |
| [HexMachina](https://arxiv.org/abs/2506.04651) | An LLM system that writes and tests a persistent Catan strategy reached 54.1% against a 51.0% depth-two AlphaBeta baseline in its reported evaluation. | This is LLM-guided code evolution, not a Transformer policy trained by gradient descent; its experiments are two-player. |

These are design references, not proof that a Transformer is the best model for
our rules. The fairest answer is a same-data, same-compute comparison against a
plain MLP.

## What is already runnable in our workspace

The local pilot found usable four-player Catanatron bots: random, weighted,
value and depth-one alpha-beta. Against three weighted bots, weighted itself
won 16/64 games (25%), value won 61/64 (95.3%), and alpha-beta depth one won
64/64. This makes value and shallow search useful first teachers, but those
numbers mostly show that weighted is weak; they do not rank the teachers
against strong play. Alpha-beta depth one is a shallow action evaluator, not a
deep-search champion.

We have now completed one from-scratch CPU Transformer pilot. It cloned the
depth-one alpha-beta action on 4,000 unique visible-state/legal-action examples
from 14 completed mixed-bot games. Its best held-out checkpoint reached 41.6%
exact action agreement on three held-out games after a 590.5-second CPU fit.
Across four disjoint board seeds with all seats rotated, it won 4/16 against
random, 3/16 against weighted, 0/16 against value, and 0/16 against depth-one
alpha-beta. The policy is legal and fast (about 0.8 ms median), but the first
cloning baseline did not reproduce the teacher's strength. Later epochs
overfit, so the epoch-8 best-loss checkpoint was retained. Three games hit the
500-turn cap and stayed in the denominator. See the [run artifacts and
results](Transformer/runs/pilot/RESULTS.md).

We also have runnable upstream ML examples, but their scores use different
setups: Charlesworth's pretrained attention policy went 32/32 against three
uniform conditional-head policies in a four-player 10-point game with trading
off; Eli's pretrained MLP went 151/256 (59%) against three heuristic bots in a
7-point game with perfect information; Benjamin's 1.38M-parameter Transformer
and graph network has no checkpoint in the inspected source and has an action
mask contract issue around Knight play. These are useful code and architecture
references, not a shared leaderboard. See the detailed [local source audit and
CPU pilot](../start/catan_lab/FINDINGS.md). The measured attention-policy
latencies were about 0.95 ms and 1.84 ms per decision on this Mac. Those runs
use different interfaces and are not an architecture bake-off, but they show
that million-parameter models can fit comfortably within inference time; data
generation and search-teacher cost need separate profiling.

## Our current rules and the gap to close

The current tournament in [`Benchmarks&Evals`](Benchmarks%26Evals/README.md)
uses four players, 10 victory points, standard discard limit, the official
spiral number placement and no friendly robber. It currently excludes
player-to-player trade offers. The benchmark still passes Catanatron's full
`game` object to the agent adapter, but our Transformer encoder reads only the
current player's private cards, public board/opponent fields, bank and legal
actions. The interface itself does not yet enforce that visibility boundary.

Before calling the final result a full Catan result, we need to settle the
actual hackathon rules and make the model-facing input an **honest observation**:
the current player's private hand and cards, public board and public opponent
information, plus the legal actions. It must not expose opponent hands or
unplayed development-card identities. If negotiation is in the target rules,
add trade offers and responses to both the action interface and evaluation.
The current Catanatron Gym wrapper is a two-player-oriented interface; the
four-player game runner is not yet a multi-agent training environment.

There is a useful negotiation design precedent: learn a small trade module
that proposes, accepts, rejects or counters offers, while the main policy
handles board actions. The 2015 four-player study used a roughly 14k-parameter
MLP for trade decisions around JSettlers. It shows that a CPU-sized specialist
can learn one hard part of Catan; it does not replace training and evaluating
the full-turn policy.

## How to make a Catan dataset with useful coverage

### Collect whole games from several kinds of players

Mix policies so the dataset does not merely copy one bot's habits:

- Random legal choices for broad, unusual state coverage.
- Weighted and victory-point bots for simple build patterns.
- Value and rollout/search bots for stronger tactical choices.
- Our own checkpoints, with some exploration, once they exist.
- Self-play games with different checkpoint versions for opponent diversity.

Run different combinations of these players. A candidate that only trains with
three copies of one opponent may exploit that opponent and then fail against a
different style.

### Store observations, choices and outcomes

Each sampled decision should keep enough raw information to replay and audit it:

- Game ID, seed, exact rules version, player seat, turn and phase.
- Board layout and port locations; roads, settlements, cities and robber.
- Current player's resources/cards, public opponent information and bank state.
- Current legal actions, the chosen action and its source policy.
- Final winner and each player's final points when the game finishes.
- Teacher name/version, search or rollout budget, and any action-value labels.
- Pattern tags and a canonical state key for deduplication.

For the pilot, keep one inspectable JSONL record per decision. A compact schema
can look like this (values are illustrative):

```json
{
  "schema_version": 1,
  "game_id": "seed-0042-game-03",
  "rules_hash": "...",
  "acting_seat": 2,
  "phase": "main_turn",
  "observation": {"board": "...", "own_hand": "...", "public_players": "..."},
  "legal_actions": ["..."],
  "policy_target": [0.0, 0.7, 0.3],
  "action_values": [null, 0.8, 0.6],
  "chosen_action": 1,
  "teacher": "catanatron_value_v0",
  "pattern_tags": ["scarce_ore", "city_vs_settlement"],
  "final_rank": 1,
  "board_seed": 42
}
```

The identity key must include every visible feature that can change legal
choices or their meaning: rules, acting player/phase, board and pieces, private
hand/cards, public opponent state, turn flags and the legal-action set. Keep
color labels player-relative. Do not merge positions just because their tile
layouts match. Store raw records first and derive tensor encodings at load time.

For the first supervised model, the key label is **which legal action the
teacher chose**. Where affordable, add a value for candidate actions and a state
value. Every position from a completed game can also carry its final outcome,
but that outcome is noisy: dice and three other players affect it. Do not treat
one win or loss as a perfect label for every decision in that game.

### Measure variety by phases and strategic patterns

Track both unique state count and the pattern mix. Useful first tags include:

| Phase or pattern | Examples to capture |
|---|---|
| Setup | Resource diversity, pip production, expansion paths, ports, seat order |
| Early game | First build, road expansion, resource shortage, settlement race |
| Mid game | City vs. settlement, development cards, port use, road blocking |
| Late game | Longest Road, Largest Army, hidden-point uncertainty, victory race |
| Dice/robber | Production on common and rare rolls, 7/discard, robber blocks or harms self |
| Economy | Bank trade, scarce resource, full hand, unused resources, development-card timing |
| Negotiation, if enabled | Offer value, accept/reject, opponent-specific trade response |

These are tags for balancing and diagnosis, not permission to merge positions
that merely look similar. A one-hex change or a different player's turn can
change the best move.

Deduplicate exact observations first. Convert each player's view to a
player-relative form so the learning problem does not depend on color names.
Only merge rotated or reflected boards after we have tested that the board,
ports, roads, vertices, action IDs and action labels all transform correctly.
Keep the raw orientation in the record. Split train/validation/test by whole
game and board seed before adding rotated/reflected training copies.

### A practical size ramp

Do not start by labeling every state from thousands of games. Use three gates:

1. **Pipeline pilot:** about 2,000 unique decision states. Inspect records,
   labels, legality, duplicates and tag coverage by hand.
2. **First useful comparison:** about 10,000–20,000 unique observations from
   mixed policies, with complete-game outcomes and a held-out game split.
3. **Expand toward 50,000+ only if useful:** add the patterns the first model
   misses and use measured CPU time to decide whether broader coverage is worth
   the cost.

These are experiment sizes, not a claim that a given number guarantees strength.
The Connect Four lab's successful pattern was to mine candidates, remove
canonical duplicates, cache teacher labels, then scale after measuring the
pilot. See its [dataset design](../start/connect_four_transformer/DATASET_PLAN.md)
and [unique-state run](../start/connect_four_transformer/STRATEGY_V5_UNIQUE_5MIN.md).

### Optional human game data

A public [Colonist.io dataset](https://github.com/OrgadYron/catan_43k_games_dataset)
describes 43,947 anonymized four-player game logs with event histories and final
results; the repository says the unpacked data is about 6.9 GB and limits use
to research and education. It could teach human action preferences and help
analyze common patterns. It is not immediately a clean Catanatron training set:
we must parse the event stream, reconstruct the player-visible state, map
actions, check rules, and group splits by game. Treat it as an optional source;
start with our own reproducible engine games. Do not download the full archive
until a small-sample parser proves useful.

## Models to compare on the same data

### Baseline: an MLP

Flatten a player-relative observation, mask illegal actions, and score each
legal choice. This is the simplest control model. If a Transformer cannot beat
it on held-out play, attention is not earning its cost yet.

### Candidate: board-aware Transformer

Represent the board as related entities rather than a long arbitrary list:

- 19 hex/tile tokens for terrain, number and robber state.
- 54 vertex tokens for buildings, adjacent production and expansion potential.
- 72 edge tokens for roads, ownership and route connectivity.
- Player tokens for the current player's private information and each
  opponent's public information; a global token for phase, dice/bank and turn.
- Board adjacency information so the model knows which hexes touch which
  vertices and which vertices are joined by edges.

Score only legal actions. A useful first design gives every candidate action a
type and destination (vertex, edge, hex or resource choice), embeds it against
the board representation, and produces one score per legal candidate. This
avoids treating every combination of action type × location × resource as an
unrelated output. Composite actions can use conditional heads, as in the public
Catan Transformer projects.

Start small: width 64–128, two or three attention blocks, four heads, then
measure. Use roughly **100k parameters** for a tiny sanity model, **300k–800k**
for the compact comparison, and include a **1.1M–1.5M** model because inspected
public Catan policies in that range run on this CPU. Build a parameter-matched
MLP control for each size. Record exact parameter count, training throughput,
inference latency and tournament strength. These are size points to measure,
not a claim that the biggest model wins. A history Transformer for trades and
opponent behavior is a later experiment; first test whether the board
representation helps.

Useful heads beyond action choice are a win/value estimate and optional
auxiliary predictions such as expected production by resource or public
opponent development. The acting policy must still use only information visible
to the acting player.

## From imitation to reinforcement learning

### Step 1: supervised learning from a random initialization

Train the MLP and Transformer from scratch to imitate legal teacher choices.
Use masked cross-entropy. If several teacher actions are similarly good, use a
soft target from their action scores or visits instead of forcing one arbitrary
answer. Track held-out teacher agreement and regret, but do not use those as the
final measure of strength.

### Step 2: PPO from the best supervised checkpoint

Pure random-start RL in four-player Catan has a very sparse signal: a game may
take many turns before a win, and three opponents influence each trajectory.
Start PPO from the learned policy, then train against a mix of fixed heuristics,
older checkpoints and self-play. Keep several frozen opponents in a league so
the model does not forget how to play against earlier styles. Give the actor
only its honest observation; a critic may use more during training only if the
final actor cannot access that hidden information.

For the first CPU PPO comparison, use the terminal win/rank as the main reward.
Add shaped rewards only as a separate experiment, because a model can learn to
maximize easy bonuses—such as building for its own sake—without winning. Track
reward components and verify that any shaping improves tournament results.

## CPU experiment plan

| Experiment | Question | Keep everything else fixed |
|---|---|---|
| A. Teacher baseline | Are the built-in heuristic/search bots strong and legal under our chosen rules? | Opponents, seeds, seats and rules |
| B. Data pilot | Do our generated states cover phases and patterns, and are labels valid? | Generator seed and teacher version |
| C. MLP vs Transformer | Does board attention improve held-out choices and tournament score? | Same rows, split, training time and parameter budget |
| D. Size sweep | Does a larger model earn its CPU cost? | Architecture, data and training budget |
| E. BC vs PPO | Does self-play improve on the supervised starting point? | Same held-out tournament and opponent suite |
| F. Mixed vs single teacher | Does policy diversity improve generalization? | Dataset size and model/training budget |

Set CPU threads to 1, 2 and 4 in a small sweep; do not assume more threads
means faster training. Record wall time, CPU time, peak memory, examples per
second and inference latency. Use at least three random initialization seeds
for architecture comparisons when compute allows. Save every run's config,
seed, dataset hash, engine commit and checkpoint hash.

The current machine is an arm64 Mac with 10 CPU cores and 24 GiB RAM, according
to the earlier Catan pilot. That is enough for compact models and lots of
simulation; it is a reason to profile teacher labeling and training separately,
not to grow the model before the baseline is measured.

## Evaluation: learning is not winning

Training/validation numbers answer whether the model learned its labels. The
tournament answers whether it plays well. Keep a separate, locked set of maps
and seeds for final evaluation.

- Main outcome: win rate and its uncertainty, with candidate rotated through
  every seat.
- Supporting outcome: final victory points, placement, turn of first expansion,
  unused resources, and resource/road/settlement/card patterns.
- Correctness: illegal actions, timeout/cap count, and hidden-information
  violations must be zero.
- Speed: median and p95 decision latency on CPU under the expected game budget.
- Robustness: test against random, weighted, value, rollout/search, prior
  checkpoints, and mixed opponent tables—not only three copies of one bot.

Use about 100 board seeds for quick screening and 500 or more for a serious
promotion check if the CPU budget permits. Four seat rotations on one board are
related games, so report uncertainty by resampling board seeds/game groups, not
pretending every seat rotation is fully independent. A new checkpoint replaces
the best only after it improves across held-out seeds and opponent mixes,
without illegal moves or a latency failure.

The current tournament is a useful screening arena, not yet a final benchmark:
it omits player offers, passes a full `game` object to the adapter, and the
first Transformer report has only four board seeds per opponent tier. Enforce
the information boundary and any required trade rules before using its scores
as a hackathon claim.

## Suggested order of work

1. Confirm the hackathon's exact rules and submission budget; freeze the
four-player base preset if the rules are not yet known.
2. Define a hidden-safe observation and enumerate every action phase, including
   setup, discard, robber and multi-choice development cards.
3. Add evaluation opponents and mixed tables; measure the built-in teachers.
4. Build a resumable mixed-policy trajectory collector, a unique-state/pattern
   report, and a teacher-label cache.
5. Train one MLP and one small board Transformer from random weights on the
   same pilot dataset; compare both in held-out games.
6. Expand the corpus toward 10k–20k unique observations where coverage is weak.
7. Add PPO league self-play only after behavior cloning produces a legal,
   competent baseline; compare it to its supervised parent.

## References

- [Catanatron source and Gym interface](https://github.com/bcollazo/catanatron)
- [Catanatron CLI and saved game outputs](https://docs.catanatron.com/using-catanatron/interactive-blocks)
- [Deep RL for four-player Catan trade dialogue](https://arxiv.org/abs/1511.08099)
- [HexMachina: LLM-guided Catan strategy code evolution](https://arxiv.org/abs/2506.04651)
- [BenjaminL1 Catan Transformer/GNN and PPO](https://github.com/BenjaminL1/catan_rl)
- [Eli6th fast Rust engine and PPO/search agent](https://github.com/Eli6th/catan-rl)
- [Charlesworth four-player PPO project and write-up](https://settlers-rl.github.io/)
- [43,947 game log dataset](https://github.com/OrgadYron/catan_43k_games_dataset)
