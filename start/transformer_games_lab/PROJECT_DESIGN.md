# Transformer Games Lab — Project Design

## Purpose

Study how transformer-based models can play games, what established research has
already explored, and what can realistically be trained or run on a CPU.

This began as a design-and-research folder. The first reusable implementation
now lives in `gamebench/`; see `README.md` for runnable adapters, tournament,
training, and CPU-profiler commands.

## Initial research findings — October 2026

### The main ways transformers appear in game research

| Family | Core idea | Games represented in the literature | What it means for this project |
|---|---|---|---|
| Move-sequence modeling | Treat moves like tokens in a sentence and predict the next move. | Go, chess, Othello | Closest first experiment: a move-history transformer for Tic-Tac-Toe. |
| Decision Transformers | Model return-to-go, state, and action as one autoregressive sequence. | Atari, OpenAI Gym, Key-to-Door; later multi-game Atari work | Useful later if we want offline reinforcement learning rather than a Minimax teacher. |
| Transformer policy/value nets with search | Use a transformer inside an AlphaZero-like policy/value + MCTS system. | Go, Hex, Gomoku, variable-size board games | Strong but far beyond the first CPU-only build. |
| Search distillation | Train a transformer to imitate values or actions produced by a powerful search engine. | Chess | This is exactly analogous to our existing MLP learning from Minimax. |
| World-model / interpretability studies | Train only on move sequences, then ask whether a board state is represented internally. | Othello, chess | A compelling second experiment after the tiny policy model. |

### Anchor papers and why they matter

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) introduced the
   transformer: a sequence model based on attention rather than recurrence.

2. [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345)
   casts offline RL as predicting actions from sequences of return-to-go, states,
   and actions. It evaluated Atari, OpenAI Gym, and Key-to-Door. This is the
   cleanest reference for a transformer that learns decisions from trajectories.

3. [The Go Transformer](https://arxiv.org/abs/2007.03500) and
   [The Chess Transformer](https://arxiv.org/abs/2008.04057) are early examples
   of treating archived game notation as language-model data.

4. [Emergent World Representations](https://arxiv.org/abs/2210.13382) trained an
   8-layer, 8-head, 512-hidden-dimension GPT on Othello move transcripts. It was
   trained to predict legal moves, not directly to win, and used probes and
   activation interventions to investigate an internal board-state representation.

5. [Multi-Game Decision Transformers](https://arxiv.org/abs/2205.15241) trained
   one offline transformer across up to 46 Atari games. It is a scale reference,
   not a CPU starter project.

6. [Amortized Planning with Large-Scale Transformers](https://arxiv.org/abs/2402.04494)
   introduced ChessBench: 10 million chess games with Stockfish-generated legal
   move and value annotations. Transformers up to 270M parameters approximate
   search-derived action values without explicit search at inference. This is the
   large-scale version of "Minimax teacher → learned neural policy."

7. [ResTNet](https://arxiv.org/abs/2410.05347) alternates convolutional and
   transformer blocks for Go and Hex. Its results highlight a practical lesson:
   board games need local spatial patterns as well as global attention, and a
   pure transformer is not automatically the strongest design.

8. [AlphaViT](https://arxiv.org/abs/2408.13871) explores transformer-based
   AlphaZero-style agents across multiple games and variable board sizes.

9. [Mastering Board Games by External and Internal Planning with Language Models](https://arxiv.org/abs/2412.12119)
   compares language-model-guided MCTS with training a model to generate a
   linearized search trace, across chess, Chess960, Connect Four, and Hex.

### Important comparisons

| Method | Learns weights? | Searches while choosing a move? | CPU suitability for our first build |
|---|---:|---:|---:|
| Minimax | No | Yes, exhaustively | Excellent for Tic-Tac-Toe teacher/evaluator |
| Q-table | Yes | No | Excellent for the existing project |
| Tiny transformer policy | Yes | No | Good learning project on CPU |
| Decision Transformer | Yes | No, but reads trajectory context | Possible only after the tiny policy works |
| MCTS + transformer | Usually | Yes, many simulations | Educational but slower and later |
| AlphaZero/MuZero scale | Yes | Yes | Not appropriate for a first CPU-only build |

### CPU implications

- A CPU can comfortably train and inspect a very small transformer on
  Tic-Tac-Toe, because game histories are short and the dataset can be generated
  exactly with Minimax.
- Standard self-attention grows roughly quadratically with sequence length.
  This barely matters for a 9-move game, but becomes important for long chess,
  Go, or Atari histories. See [On the Computational Complexity of Self-Attention](https://arxiv.org/abs/2209.04881).
- CPU feasibility is controlled by three things: parameter count, sequence
  length, and dataset size. The 270M-parameter ChessBench study and the
  20-million-game Othello study are research-scale references, not laptop CPU
  targets.
- Our CPU-first target should therefore be explanatory, not state of the art:
  a tiny transformer that is fast enough to train, inspect attention from, and
  benchmark honestly against perfect Minimax play.

### Recommended first experiment

```text
Game:              Tic-Tac-Toe
Input tokens:       move history, including player and square
Sequence length:    at most 9 moves
Teacher:            exact Minimax optimal moves
Model role:         next-move policy, not search
Evaluation:         legal-move rate, optimal-move rate, and games against Minimax
Compute target:     CPU-only, short reproducible training run
Interpretability:   display attention from the predicted move to earlier moves
```

This experiment will answer the key question cleanly: what does attention add
when a board state can already be handled by a small MLP?

## Questions this project will answer

1. What does it mean for a transformer to play a game?
2. Which games have appeared in transformer-game research and why?
3. How do transformers compare with Minimax, MCTS, Q-learning, and neural
   policies such as the Tic-Tac-Toe MLP we already built?
4. Which transformer experiments are feasible on a regular CPU?
5. What is the smallest transformer game project worth building from scratch?

## Planned research tracks

### 1. Game-transformer literature map

Create a readable map of important work on sequence models and games. The later
research will separate:

- board games and perfect-information games;
- game-playing agents trained from interaction;
- offline reinforcement learning from game trajectories;
- language-model agents that reason about games;
- transformer world models and planning models.

Each entry should record the game, model type, training data, compute used,
evaluation method, and main idea.

### 2. Core concepts notebook

Explain the ideas from first principles before implementing anything:

- game state as tokens;
- move history as a sequence;
- attention over earlier moves;
- policy prediction: which move to play next;
- value prediction: how likely a position is to win;
- supervised learning versus self-play reinforcement learning;
- transformer policy versus Minimax and MCTS.

### 3. CPU feasibility study

Measure what is practical without a GPU:

- tiny transformers trained on short Tic-Tac-Toe trajectories;
- parameter count, sequence length, RAM, and training time;
- inference latency for one legal move;
- when NumPy is enough and when PyTorch is helpful;
- the point where CPU experimentation stops being useful.

### 4. Smallest build candidate

The first candidate should be deliberately small:

```text
Input:  a sequence of Tic-Tac-Toe moves or board tokens
Output: probabilities for the next legal move
Teacher: Minimax-generated optimal games
Runtime: CPU only
```

This will make the relationship to the existing MLP concrete: the MLP sees a
board all at once, while a transformer sees a sequence and uses attention to
decide which earlier moves matter.

## Current folder layout

```text
transformer_games_lab/
├── gamebench/
│   ├── games/              # OpenSpiel-like adapters
│   ├── agents/             # random, greedy, and Transformer policies
│   ├── models/             # configurable causal Transformer
│   ├── search/             # minimax and MCTS
│   ├── training/           # initial imitation-learning path
│   ├── evaluation/         # tournaments and Elo
│   └── systems/            # pluggable CPU benchmark kernel/workloads
├── tests/                  # rules, model, tournament, profiler tests
├── train.py                # stable training entry point
├── tournament.py           # stable evaluation entry point
├── benchmark.py            # parameter-grid CPU profiler
├── simd_benchmark.py       # native scalar versus SIMD experiment
├── matrix_benchmark.py     # dtype/shape/thread GEMM experiment
├── hardware_report.py      # CPU ISA and library-backend inventory
├── dashboard.py            # convenience launcher for the Next app
├── web/                    # sole browser app: profiler + learning routes
├── SYSTEMS_LAB.md          # SIMD, AVX, SME/AMX learning notes
├── pyproject.toml          # reproducible package/dependencies
├── README.md               # commands and benchmark methodology
└── PROJECT_DESIGN.md       # research context and roadmap
```

## Learning sequence

1. Learn transformers on paper using a tiny next-token example.
2. Research the game-transformer literature with cited sources.
3. Build a CPU-friendly transformer that predicts Tic-Tac-Toe moves.
4. Compare it with the saved Q-table, MLP, Minimax, and MCTS.
5. Visualize attention, predicted moves, compute cost, and failure cases.

## Success criteria

- Explain every new idea using a small hand-worked example before code.
- Keep the first model small enough to train and inspect on CPU.
- Use a clean benchmark against Minimax, not only self-play results.
- Report model size, training time, memory assumptions, and accuracy.
- Be explicit about what the transformer learned versus what search solves.

## Decisions deferred until research begins

- Exact papers and literature selection.
- Dataset format and number of generated games.
- Tokenization scheme for moves versus board states.
- Framework choice: NumPy-only, PyTorch CPU, or both.
- Transformer width, depth, heads, learning rate, and training schedule.
- Whether MCTS is a comparison only or part of a hybrid agent.
