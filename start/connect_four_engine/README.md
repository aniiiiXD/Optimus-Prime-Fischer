# Connect Four engine

This folder contains only the deterministic game foundation. It has no neural
network, search policy, training loop, or third-party dependency yet.

## Rules encoded

- Board: 6 rows × 7 columns.
- Players alternate dropping one piece into a non-full column.
- Gravity places the piece in the lowest empty cell.
- Four horizontal, vertical, or diagonal pieces win immediately.
- A completely filled board without a winner is a draw.

## Representation contract

| Concept | Representation |
|---|---|
| empty / Player 1 / Player 2 | `0 / 1 / -1` |
| board | flat 42-value row-major list, top row first |
| engine action | zero-based column `0..6` |
| terminal UI action | human-facing column `1..7` |
| relative encoding | own piece `1`, opponent `-1`, empty `0` |
| cached state | `(tuple(board), current_player)` |

The mutable engine supports `step()` and O(1) `undo()`, which lets a future
Minimax or alpha-beta search explore a branch without copying all 42 cells.
`canonical_key()` folds horizontally mirrored positions into the same cache
entry.

## API

```python
from connect_four_engine import ConnectFour

game = ConnectFour()
game.step(3)                    # Player 1 drops in the center column
print(game.legal_actions())     # (0, 1, 2, 3, 4, 5, 6)
state = game.snapshot()         # immutable dataset/logging record
cache_key = game.canonical_key()
game.undo()
```

## Run

From the workspace root:

```bash
python3 -m unittest connect_four_engine.test_engine -v
python3 -m connect_four_engine
```

## Next layer

Once this rules layer is frozen, the next module should analyze branching
factor, tactical motifs, recursive alpha-beta search, transposition-table hit
rates, and position sampling before we choose Transformer tokens. The CPU
Transformer and dataset plan is in
[`../connect_four_transformer/PLAN.md`](../connect_four_transformer/PLAN.md).
