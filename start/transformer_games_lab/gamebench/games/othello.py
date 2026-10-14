"""Dependency-free Othello adapter with explicit pass action 64."""

from __future__ import annotations

from dataclasses import dataclass, field

from gamebench.games.base import GameAdapter, GameSpec, GameState, ObservationSpec


SIZE = 8
PASS_ACTION = 64
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
OTHELLO_SPEC = GameSpec(
    name="othello",
    num_players=2,
    action_space_size=65,
    # At most 60 placements plus interspersed forced passes. Consecutive passes
    # end the game, so 120 is a conservative, explicit sequence bound.
    max_game_length=120,
    observation=ObservationSpec(
        shape=(3, SIZE, SIZE),
        dtype="float32",
        semantics="current-player discs, opponent discs, side-to-move plane",
    ),
)


def _initial_board() -> list[int]:
    board = [-1] * 64
    board[3 * 8 + 3] = 1
    board[3 * 8 + 4] = 0
    board[4 * 8 + 3] = 0
    board[4 * 8 + 4] = 1
    return board


@dataclass
class OthelloState(GameState):
    board: list[int] = field(default_factory=_initial_board)
    _current_player: int = 0
    _history: list[int] = field(default_factory=list)
    consecutive_passes: int = 0
    spec: GameSpec = OTHELLO_SPEC

    @property
    def current_player(self) -> int:
        return self._current_player

    @property
    def history(self) -> tuple[int, ...]:
        return tuple(self._history)

    def _flips(self, action: int, player: int) -> list[int]:
        if not 0 <= action < 64 or self.board[action] != -1:
            return []
        row, col = divmod(action, 8)
        result: list[int] = []
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            ray: list[int] = []
            while 0 <= r < 8 and 0 <= c < 8 and self.board[r * 8 + c] == 1 - player:
                ray.append(r * 8 + c)
                r += dr
                c += dc
            if ray and 0 <= r < 8 and 0 <= c < 8 and self.board[r * 8 + c] == player:
                result.extend(ray)
        return result

    def _placements(self, player: int) -> tuple[int, ...]:
        return tuple(action for action in range(64) if self._flips(action, player))

    def legal_actions(self) -> tuple[int, ...]:
        if self.is_terminal():
            return ()
        placements = self._placements(self._current_player)
        return placements if placements else (PASS_ACTION,)

    def apply_action(self, action: int) -> None:
        legal = self.legal_actions()
        if action not in legal:
            raise ValueError(f"illegal Othello action: {self.action_name(action)}")
        if action == PASS_ACTION:
            self.consecutive_passes += 1
        else:
            flips = self._flips(action, self._current_player)
            self.board[action] = self._current_player
            for square in flips:
                self.board[square] = self._current_player
            self.consecutive_passes = 0
        self._history.append(action)
        self._current_player = 1 - self._current_player

    def is_terminal(self) -> bool:
        if self.consecutive_passes >= 2 or all(cell != -1 for cell in self.board):
            return True
        return not self._placements(0) and not self._placements(1)

    def returns(self) -> tuple[float, float]:
        if not self.is_terminal():
            return (0.0, 0.0)
        difference = self.board.count(0) - self.board.count(1)
        if difference == 0:
            return (0.0, 0.0)
        return (1.0, -1.0) if difference > 0 else (-1.0, 1.0)

    def observation_tensor(self, player: int | None = None) -> tuple[float, ...]:
        viewer = self._current_player if player is None else player
        own = tuple(float(cell == viewer) for cell in self.board)
        opponent = tuple(float(cell == 1 - viewer) for cell in self.board)
        side = tuple([1.0 if self._current_player == viewer else 0.0] * 64)
        return own + opponent + side

    def clone(self) -> "OthelloState":
        return OthelloState(self.board.copy(), self._current_player, self._history.copy(), self.consecutive_passes)

    def action_name(self, action: int) -> str:
        if action == PASS_ACTION:
            return "pass"
        if not 0 <= action < 64:
            return str(action)
        row, col = divmod(action, 8)
        return f"{'abcdefgh'[row]}{col + 1}"

    def render(self) -> str:
        glyph = {-1: ".", 0: "●", 1: "○"}
        rows = [f"{'abcdefgh'[row]} " + " ".join(glyph[self.board[row * 8 + col]] for col in range(8)) for row in range(8)]
        return "  1 2 3 4 5 6 7 8\n" + "\n".join(rows)


class OthelloAdapter(GameAdapter):
    spec = OTHELLO_SPEC

    def reset(self, seed: int | None = None) -> OthelloState:
        del seed
        return OthelloState()
