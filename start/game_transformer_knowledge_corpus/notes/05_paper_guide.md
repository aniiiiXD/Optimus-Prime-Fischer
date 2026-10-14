# Paper guide

| Local PDF | Main question | What to extract for our project |
|---|---|---|
| `01_mastering_board_games_external_internal_planning.pdf` | how language models combine internal prediction with external planning in board games | separate learned policy/world knowledge from explicit search |
| `02_emergent_world_representations_othello_gpt.pdf` | whether next-token models learn the hidden state that generates sequences | history tokenization, causal GPT, probes, and interventions |
| `03_chess_transformer.pdf` | how a Transformer can model chess games as sequences | chess tokenization, training objective, scale, and evaluation |
| `04_go_transformer.pdf` | how Transformer sequence modeling transfers to Go | large action spaces and long spatial dependencies |
| `05_chessformer_unified_architecture.pdf` | how one architecture can cover move prediction and related chess tasks | multi-task design and reusable representation |
| `06_mastering_chess_transformer.pdf` | how supervised Transformer training reaches strong chess play | dataset curation, policy quality, and evaluation methodology |
| `07_word_play_othello.pdf` | how language-model machinery can model Othello play | alternative framing of moves as a sequence language |

## Questions to answer while reading

For every paper, record:

1. What exactly is one token?
2. Is the input history, current state, or both?
3. Does the output predict one move, every legal move, value, or text?
4. Where is legality enforced?
5. What generated the targets: humans, random play, engine search, or self-play?
6. What is the model size and compute budget?
7. Is strength measured by loss, move accuracy, legal accuracy, Elo, or win rate?
8. Which claim is demonstrated, and which claim is merely inferred?

## Reading warning

Do not compare reported “accuracy” across papers until the denominator is clear. Predicting the exact human move, predicting any legal move, reconstructing a board square, and winning a game are different tasks.
