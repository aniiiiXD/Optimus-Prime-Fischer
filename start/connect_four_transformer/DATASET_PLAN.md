# Connect Four dataset plan for a CPU

## The central choice

We will not enumerate all Connect Four boards or label every state with a
perfect solver. That is unnecessary and too expensive for our first model.
Instead, we will generate a small, diverse set of legal positions and ask a
bounded search teacher to score every legal move in each position.

The corpus contains much larger precedents: the downloaded
[Connect Four report](../game_transformer_knowledge_corpus/repos/connect-4/REPORT.md)
describes about 700k MCTS positions with 2,000 rollouts per decision, and the
[MAV takeaway](../game_transformer_knowledge_corpus/notes/06_mav_takeaway.md)
reports a much larger solver-labeled Connect Four corpus.
Those show that search targets are useful; they do not define the minimum size
or CPU cost for our experiment.

## Dataset stages

### Pilot: prove the generator

Create **1,000–2,000 unique canonical positions** first. For each, record how
long the search teacher takes at two small node budgets. Compare its choices
with the tactical baseline and inspect errors before scaling.

### First useful dataset

Target **10,000–20,000 unique canonical positions**, only if the pilot shows
the labels are useful and affordable. Keep the first release compact and
reproducible. Expand in later versions rather than spending the CPU budget
before we know which positions matter.

### How positions are generated

Generate complete legal games with a mixture of players so the dataset does
not simply reproduce one policy's favorite openings:

| Source games | Approximate share | Purpose |
|---|---:|---|
| Random / center-biased legal policies | 30% | Broad coverage and varied openings |
| Tactical heuristic versus itself or other policies | 30% | Immediate wins, blocks, forks and gravity traps |
| Shallow alpha-beta self-play at varied budgets | 40% | Strategic states and positions near a stronger policy's decisions |

Alternate which side starts each game. Vary the first several moves and sample
positions throughout the game. Use a stratified target by move count so the
empty-board opening does not dominate: approximately 20% plies 0–8, 30% plies
9–18, 30% plies 19–30, and 20% plies 31–41. Track tactical tags separately;
make sure immediate wins, forced blocks, double threats, and support-square
traps appear in every phase where they can occur.

When two player policies are available, add controlled exploration: with a
small configured probability, sample among near-best legal actions instead of
always taking the top action. Record that probability and the policy versions
so the data can be reproduced.

### How each state is labeled

Run one bounded alpha-beta search at the position and retain the root result
for **every legal column**:

- `Q(s,a)`: search estimate after dropping in column `a`, from the current player's perspective;
- `V(s)`: best root value from the same perspective;
- `policy_target`: probability mass on best or near-best moves, derived from root scores;
- `outcome`: final win/draw/loss from the trajectory, when the game was completed;
- `teacher_depth`, `nodes`, and `search_score`: how much computation produced the labels.

At terminal wins, use exact `+1` or `-1`. At nonterminal cutoffs, keep the
bounded-search evaluation as a separate target from the eventual game result;
do not pretend a heuristic leaf score is a perfect outcome. The first policy
target can distribute probability evenly across score-tied best moves. Once
we add MCTS, use normalized root visit counts as a richer soft policy target.

The search teacher starts with immediate-win and forced-block checks, then
negamax alpha-beta with center-first move ordering and a transposition table.
Search budget is capped and logged. If Python search throughput is too low,
profile the rules/search boundary before scaling labels; a compact bitboard
search can replace the teacher without changing the dataset schema.

## Deduplication, symmetry, and leakage control

1. Key positions by the engine's horizontal-mirror canonical state plus side to move.
2. If mirrored positions are encountered, transform the seven action labels into the canonical orientation before merging them.
3. Store one canonical record and accumulate teacher visits/scores only when labels come from compatible teacher settings.
4. Split by opening/game groups and canonical state **before** augmentation; no exact or mirrored state may appear in both train and test.
5. Apply random horizontal reflection to training examples only. Reverse legal columns, policy probabilities, Q values, and visits; leave the scalar value and final result unchanged.
6. Keep a separate tactical puzzle suite and fixed tournament openings outside training and validation.

## Record format

One JSONL record per state during the inspectable pilot; move to compressed
NumPy shards for larger training runs.

```json
{
  "schema_version": 1,
  "board": "42 cell values in row-major order",
  "current_player": 1,
  "ply": 18,
  "legal_actions": [0, 1, 2, 3, 4, 5, 6],
  "landing_rows": [5, 5, 5, 5, 5, 5, 5],
  "policy_target": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
  "action_values": [null, null, null, 0.7, null, null, null],
  "state_value": 0.7,
  "trajectory_result": 1,
  "teacher": "negamax_ab",
  "teacher_depth": 6,
  "teacher_nodes": 1200,
  "source_policy": "mixed_selfplay_v0",
  "game_id": 17,
  "opening_group": "seeded-prefix-id"
}
```

The values above are illustrative. Actual `action_values` must be filled for
each legal action, and `landing_rows` must match the current board. Store the
raw board and metadata, then derive model tensors during loading so encodings
can change without regenerating search labels.

## CPU budget and stop/go rule

- Start with a **100-position timing probe** at several node budgets.
- Report median and p95 labeling time, nodes/second, cache hit rate, and teacher agreement with the tactical baseline.
- Use that measured rate to estimate pilot and full-set time before launching generation.
- Generate in resumable chunks with a fixed seed and checkpoint the canonical-state index.
- Keep generation, search labeling, and model training profiles separate.
- Only move from 2k to 10k+ states when the pilot's tactical and held-out results justify the extra search.

This prevents an expensive dataset from becoming a giant collection of weak or
duplicated targets. A few thousand informative positions are enough to test
whether our encoding and training loop work; strength claims require the later
held-out games and repeated seeds.
