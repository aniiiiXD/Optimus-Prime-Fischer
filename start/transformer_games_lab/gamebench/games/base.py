"""The only interface a new game must implement.

The names deliberately resemble OpenSpiel: current_player, legal_actions,
apply_action, is_terminal, returns, and observation_tensor.  We do not depend
on OpenSpiel, but adapters can be wrapped in either direction later.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


PAD_TOKEN = 0
BOS_TOKEN = 1
ACTION_TOKEN_OFFSET = 2


@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple[int, ...]
    dtype: str
    semantics: str


@dataclass(frozen=True)
class GameSpec:
    name: str
    num_players: int
    action_space_size: int
    max_game_length: int
    observation: ObservationSpec
    information: str = "perfect"

    @property
    def token_vocab_size(self) -> int:
        """PAD + BOS + one token for every action."""
        return self.action_space_size + ACTION_TOKEN_OFFSET


class GameState(ABC):
    """Mutable state with explicit clone semantics.

    Agents receive a clone during tournaments, so an agent cannot corrupt the
    authoritative match state.
    """

    spec: GameSpec

    @property
    @abstractmethod
    def current_player(self) -> int:
        """Player index that must act next."""

    @property
    @abstractmethod
    def history(self) -> tuple[int, ...]:
        """Actions taken so far, in order."""

    @abstractmethod
    def legal_actions(self) -> tuple[int, ...]:
        """Legal actions for current_player."""

    @abstractmethod
    def apply_action(self, action: int) -> None:
        """Mutate this state by applying one legal action."""

    @abstractmethod
    def is_terminal(self) -> bool:
        """Whether no further action can be played."""

    @abstractmethod
    def returns(self) -> tuple[float, ...]:
        """Terminal utility for every player; zero-sum games use +1/-1/0."""

    @abstractmethod
    def observation_tensor(self, player: int | None = None) -> tuple[float, ...]:
        """Fixed-size numeric observation, with semantics declared by spec."""

    @abstractmethod
    def clone(self) -> "GameState":
        """Independent copy suitable for search."""

    def history_tokens(self) -> tuple[int, ...]:
        """Generic autoregressive representation: BOS then shifted actions."""
        return (BOS_TOKEN, *(action + ACTION_TOKEN_OFFSET for action in self.history))

    def action_to_token(self, action: int) -> int:
        if not 0 <= action < self.spec.action_space_size:
            raise ValueError(f"action {action} is outside the action space")
        return action + ACTION_TOKEN_OFFSET

    def token_to_action(self, token: int) -> int:
        action = token - ACTION_TOKEN_OFFSET
        if not 0 <= action < self.spec.action_space_size:
            raise ValueError(f"token {token} is not an action token")
        return action

    def action_name(self, action: int) -> str:
        return str(action)

    def render(self) -> str:
        return repr(self)


class GameAdapter(ABC):
    spec: GameSpec

    @abstractmethod
    def reset(self, seed: int | None = None) -> GameState:
        """Create a fresh initial state."""

    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "players": self.spec.num_players,
            "action_space": self.spec.action_space_size,
            "token_vocabulary": self.spec.token_vocab_size,
            "max_game_length": self.spec.max_game_length,
            "observation_shape": self.spec.observation.shape,
            "information": self.spec.information,
        }
