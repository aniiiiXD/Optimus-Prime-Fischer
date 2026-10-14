"""Connect Four adapter: seven column actions, six-by-seven board."""

from __future__ import annotations

from dataclasses import dataclass, field

from gamebench.games.base import GameAdapter, GameSpec, GameState, ObservationSpec


ROWS, COLS = 6, 7
CONNECT4_SPEC = GameSpec(
    name="connect4",
    num_players=2,
    action_space_size=COLS,
    max_game_length=ROWS * COLS,
    observation=ObservationSpec(
        shape=(3, ROWS, COLS),
        dtype="float32",
        semantics="player-0 stones, player-1 stones, side-to-move plane",
    ),
)


@dataclass
class ConnectFourState(GameState):
    board: list[int] = field(default_factory=lambda: [-1] * (ROWS * COLS))
    _current_player: int = 0
    _history: list[int] = field(default_factory=list)
    _winner: int | None = None
    spec: GameSpec = CONNECT4_SPEC

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def history(self) -> tuple[int, ...]:
        return tuple(self._history)

    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal():
            return ()
        return tuple(column for column in range(COLS) if self.board[column] == -1)

    def apply_action(self, action: int) -> None:
        if action not in self.legal_actions():
            raise ValueError(f"illegal Connect Four column: {action}")
        row = next(row for row in range(ROWS - 1, -1, -1) if self.board[row * COLS + action] == -1)
        self.board[row * COLS + action] = self._current_player
        self._history.append(action)
        if self._has_four(row, action, self._current_player):
            self._winner = self._current_player
        self._current_player = 1 - self._current_player

    def _has_four(self, row: int, col: int, player: int) -> bool:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (-1, 1):
                r, c = row + sign * dr, col + sign * dc
                while 0 <= r < ROWS and 0 <= c < COLS and self.board[r * COLS + c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= 4:
                return True
        return False

    def is_terminal(self) -> bool:
        return self._winner is not None or len(self._history) == ROWS * COLS

    def returns(self) -> tuple[float, float]:
        if not self.is_terminal():
            return (0.0, 0.0)
        if self._winner is None:
            return (0.0, 0.0)
        return (1.0, -1.0) if self._winner == 0 else (-1.0, 1.0)

    def observation_tensor(self, player: int | None = None) -> tuple[float, ...]:
        viewer = self._current_player if player is None else player
        first = tuple(float(cell == viewer) for cell in self.board)
        second = tuple(float(cell == 1 - viewer) for cell in self.board)
        side = tuple([1.0 if self._current_player == viewer else 0.0] * (ROWS * COLS))
        return first + second + side

    def clone(self) -> "ConnectFourState":
        return ConnectFourState(self.board.copy(), self._current_player, self._history.copy(), self._winner)

    def action_name(self, action: int) -> str:
        return f"column {action + 1}"

    def render(self) -> str:
        glyph = {-1: ".", 0: "X", 1: "O"}
        rows = [" ".join(glyph[self.board[row * COLS + col]] for col in range(COLS)) for row in range(ROWS)]
        return "1 2 3 4 5 6 7\n" + "\n".join(rows)


class ConnectFourAdapter(GameAdapter):
    spec = CONNECT4_SPEC

    def reset(self, seed: int | None = None) -> ConnectFourState:
        del seed
        return ConnectFourState()
