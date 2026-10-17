#!/usr/bin/env python3
"""Fine-tune the legal-action MLP with terminal-win PPO self-play.

The policy only scores actions supplied by Catanatron, so it cannot emit an
illegal action.  The only reward is the eventual game result: +1 win, -1 loss,
and 0 for a turn-cap draw.  There is deliberately no hand-written Catan reward.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from train_legal_ranker import HERE, LegalRanker, encode_action, encode_state, stable_edge_index


WORKSPACE = HERE.parents[2]
CATANATRON_ROOT = WORKSPACE / "start/catan_lab/vendor/catanatron/catanatron"
sys.path.insert(0, str(CATANATRON_ROOT))
sys.path.insert(0, str(WORKSPACE))

from catanatron import Color, Game, RandomPlayer  # noqa: E402
from catanLabs.Catanatron.tournament_protocol import install_v5_protocol  # noqa: E402
from generate_dataset import action_json, make_bot, observation  # noqa: E402


class ActorCriticRanker(nn.Module):
    """Pretrained legal-action policy plus a state-value critic, both MLPs."""

    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        # This exactly matches the imitation learner, making its checkpoint a
        # valid policy initialization for PPO.
        self.policy = LegalRanker(state_size, action_size)
        self.critic = nn.Sequential(
            nn.Linear(state_size, 384), nn.ReLU(), nn.LayerNorm(384),
            nn.Linear(384, 192), nn.ReLU(), nn.Linear(192, 1),
        )

    def policy_logits(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.policy(state, actions)

    def value(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


def tensors_for_position(
    game: Game,
    color: Color,
    legal: list[Any],
    edge_index: dict[tuple[int, int], int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    visible = observation(game, color)
    state = torch.tensor([encode_state(visible, edge_index)], dtype=torch.float32, device=device)
    actions = torch.tensor(
        [[encode_action(action_json(action, color), visible, edge_index) for action in legal]],
        dtype=torch.float32,
        device=device,
    )
    return state, actions


class FrozenRankerPlayer:
    """An immutable past-policy opponent used to prevent self-play collapse."""

    def __init__(
        self,
        color: Color,
        policy_state: dict[str, torch.Tensor],
        state_size: int,
        action_size: int,
        edge_index: dict[tuple[int, int], int],
    ) -> None:
        self.color, self.edge_index = color, edge_index
        self.model = LegalRanker(state_size, action_size)
        self.model.load_state_dict(policy_state)
        self.model.eval()

    @torch.no_grad()
    def decide(self, game: Game, playable_actions: list[Any]) -> Any:
        state, actions = tensors_for_position(
            game, self.color, list(playable_actions), self.edge_index, torch.device("cpu")
        )
        return playable_actions[int(self.model(state, actions).argmax(dim=1).item())]


class OpponentPool:
    """Mixes external bots with frozen policy snapshots."""

    def __init__(
        self,
        codes: list[str],
        state_size: int,
        action_size: int,
        edge_index: dict[tuple[int, int], int],
        seed: int,
    ) -> None:
        allowed = {"F_BASE", "F_CONTENDER", "VP", "W", "R", "SELF"}
        invalid = set(codes) - allowed
        if invalid:
            raise ValueError(f"Unsupported opponent codes: {sorted(invalid)}")
        self.codes = codes
        self.state_size, self.action_size, self.edge_index = state_size, action_size, edge_index
        self.rng = random.Random(seed)
        self.snapshots: deque[dict[str, torch.Tensor]] = deque(maxlen=8)

    def add_snapshot(self, policy: LegalRanker) -> None:
        self.snapshots.append({key: value.detach().cpu().clone() for key, value in policy.state_dict().items()})

    def make(self, color: Color) -> tuple[str, Any]:
        code = self.rng.choice(self.codes)
        if code != "SELF":
            return code, make_bot(code, color)
        # The initial snapshot is installed before the first game.
        snapshot = self.rng.choice(list(self.snapshots))
        return code, FrozenRankerPlayer(color, snapshot, self.state_size, self.action_size, self.edge_index)


class CatanSelfPlayEnv:
    """One-agent environment: opponent turns are executed internally."""

    def __init__(
        self,
        pool: OpponentPool,
        edge_index: dict[tuple[int, int], int],
        seed: int,
        vps_to_win: int,
        max_plies: int,
    ) -> None:
        self.pool, self.edge_index = pool, edge_index
        self.seed, self.vps_to_win, self.max_plies = seed, vps_to_win, max_plies
        self.game: Game | None = None
        self.agent_color: Color | None = None
        self.opponent: Any = None
        self.plies = 0
        self.opponent_name = ""

    def _advance_to_agent(self) -> bool:
        """Play only opponent moves; return whether the game ended."""
        assert self.game is not None and self.agent_color is not None
        while self.game.winning_color() is None and self.plies < self.max_plies:
            if self.game.state.current_color() == self.agent_color:
                return False
            legal = list(self.game.playable_actions)
            action = self.opponent.decide(self.game, legal)
            if action not in legal:
                raise RuntimeError(f"Opponent {self.opponent_name} selected an illegal action")
            self.game.execute(action, validate_action=True)
            self.plies += 1
        return True

    def reset(self, episode: int) -> tuple[Game, Color, list[Any], bool]:
        colors = (Color.BLUE, Color.RED)
        self.game = Game([RandomPlayer(color) for color in colors], seed=self.seed + episode, vps_to_win=self.vps_to_win)
        self.agent_color = self.game.state.colors[episode % 2]
        opponent_color = next(color for color in colors if color != self.agent_color)
        self.opponent_name, self.opponent = self.pool.make(opponent_color)
        self.plies = 0
        done = self._advance_to_agent()
        return self.game, self.agent_color, list(self.game.playable_actions), done

    def step(self, action: Any) -> tuple[list[Any], bool, float]:
        assert self.game is not None and self.agent_color is not None
        legal = list(self.game.playable_actions)
        if action not in legal:
            raise RuntimeError("PPO policy selected an illegal action")
        self.game.execute(action, validate_action=True)
        self.plies += 1
        done = self._advance_to_agent()
        winner = self.game.winning_color()
        if winner is None:
            reward = 0.0
        else:
            reward = 1.0 if winner == self.agent_color else -1.0
        return list(self.game.playable_actions), done, reward


@dataclass
class Transition:
    state: torch.Tensor
    actions: torch.Tensor
    action_index: int
    old_logprob: float
    old_value: float
    reward: float
    done: bool
    advantage: float = 0.0
    return_: float = 0.0


@torch.no_grad()
def collect_episode(
    model: ActorCriticRanker,
    env: CatanSelfPlayEnv,
    episode: int,
    edge_index: dict[tuple[int, int], int],
    device: torch.device,
    temperature: float,
) -> tuple[list[Transition], dict[str, Any]]:
    game, color, legal, done = env.reset(episode)
    trajectory: list[Transition] = []
    while not done:
        state, actions = tensors_for_position(game, color, legal, edge_index, device)
        logits = model.policy_logits(state, actions).squeeze(0) / temperature
        distribution = Categorical(logits=logits)
        action_index = int(distribution.sample().item())
        chosen = legal[action_index]
        next_legal, done, reward = env.step(chosen)
        trajectory.append(
            Transition(
                state=state.cpu(), actions=actions.cpu(), action_index=action_index,
                old_logprob=float(distribution.log_prob(torch.tensor(action_index, device=device)).item()),
                old_value=float(model.value(state).item()), reward=reward, done=done,
            )
        )
        legal = next_legal
    winner = env.game.winning_color() if env.game is not None else None
    return trajectory, {
        "won": winner == color,
        "draw": winner is None,
        "plies": env.plies,
        "agent_decisions": len(trajectory),
        "opponent": env.opponent_name,
    }


def add_returns_and_advantages(trajectory: list[Transition], gamma: float, gae_lambda: float) -> None:
    gae = 0.0
    next_value = 0.0
    for transition in reversed(trajectory):
        nonterminal = 0.0 if transition.done else 1.0
        delta = transition.reward + gamma * next_value * nonterminal - transition.old_value
        gae = delta + gamma * gae_lambda * nonterminal * gae
        transition.advantage = gae
        transition.return_ = gae + transition.old_value
        next_value = transition.old_value


def ppo_update(
    model: ActorCriticRanker,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    device: torch.device,
    ppo_epochs: int,
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_grad_norm: float,
    temperature: float,
) -> dict[str, float]:
    advantages = torch.tensor([item.advantage for item in transitions], dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    policy_losses, value_losses, entropies = [], [], []
    for _ in range(ppo_epochs):
        order = torch.randperm(len(transitions)).tolist()
        for index in order:
            item = transitions[index]
            state, actions = item.state.to(device), item.actions.to(device)
            logits = model.policy_logits(state, actions).squeeze(0) / temperature
            distribution = Categorical(logits=logits)
            action_index = torch.tensor(item.action_index, device=device)
            new_logprob = distribution.log_prob(action_index)
            ratio = torch.exp(new_logprob - item.old_logprob)
            advantage = advantages[index].to(device)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
            policy_loss = -torch.minimum(unclipped, clipped)
            value_loss = F.mse_loss(model.value(state), torch.tensor([item.return_], device=device))
            entropy = distribution.entropy()
            loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.detach()))
            value_losses.append(float(value_loss.detach()))
            entropies.append(float(entropy.detach()))
    return {
        "policy_loss": round(sum(policy_losses) / len(policy_losses), 5),
        "value_loss": round(sum(value_losses) / len(value_losses), 5),
        "entropy": round(sum(entropies) / len(entropies), 5),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init-checkpoint", type=Path, default=HERE / "runs/legal_ranker_v1/model.pt")
    parser.add_argument("--output", type=Path, default=HERE / "runs/ppo_actor_critic_v1")
    parser.add_argument("--updates", type=int, default=4)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.20)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--opponents", default="SELF,F_BASE,VP,W,R")
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20264011)
    parser.add_argument("--vps-to-win", type=int, default=10)
    parser.add_argument("--max-plies", type=int, default=10_000)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    args = parser.parse_args()
    if min(args.updates, args.episodes_per_update, args.ppo_epochs, args.snapshot_every) < 1:
        parser.error("update, episode, epoch, and snapshot counts must be positive")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")

    install_v5_protocol()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu")
    checkpoint = torch.load(args.init_checkpoint, map_location="cpu", weights_only=True)
    with (HERE / "data/catanatron_diverse.jsonl").open(encoding="utf-8") as handle:
        first = json.loads(next(handle))
    edge_index = stable_edge_index(first["observation"])
    model = ActorCriticRanker(checkpoint["state_size"], checkpoint["action_size"]).to(device)
    model.policy.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    pool = OpponentPool(args.opponents.split(","), checkpoint["state_size"], checkpoint["action_size"], edge_index, args.seed)
    pool.add_snapshot(model.policy)
    env = CatanSelfPlayEnv(pool, edge_index, args.seed, args.vps_to_win, args.max_plies)
    args.output.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}

    history = []
    episode_index = 0
    for update in range(1, args.updates + 1):
        model.eval()
        transitions: list[Transition] = []
        summaries = []
        for _ in range(args.episodes_per_update):
            trajectory, summary = collect_episode(model, env, episode_index, edge_index, device, args.temperature)
            add_returns_and_advantages(trajectory, args.gamma, args.gae_lambda)
            transitions.extend(trajectory)
            summaries.append(summary)
            episode_index += 1
        model.train()
        losses = ppo_update(
            model, optimizer, transitions, device, args.ppo_epochs, args.clip_ratio,
            args.value_coefficient, args.entropy_coefficient, args.max_grad_norm,
            args.temperature,
        )
        opponents = Counter(summary["opponent"] for summary in summaries)
        metric = {
            "update": update,
            "episodes": len(summaries),
            "transitions": len(transitions),
            "wins": sum(summary["won"] for summary in summaries),
            "draws": sum(summary["draw"] for summary in summaries),
            "win_rate": round(sum(summary["won"] for summary in summaries) / len(summaries), 3),
            "mean_agent_decisions": round(sum(summary["agent_decisions"] for summary in summaries) / len(summaries), 1),
            "mean_plies": round(sum(summary["plies"] for summary in summaries) / len(summaries), 1),
            "opponents": dict(opponents),
            **losses,
        }
        history.append(metric)
        print(json.dumps(metric))
        if update % args.snapshot_every == 0:
            pool.add_snapshot(model.policy)
        torch.save(
            {
                "state_size": checkpoint["state_size"], "action_size": checkpoint["action_size"],
                "policy": model.policy.state_dict(), "critic": model.critic.state_dict(),
                "update": update, "config": config,
            },
            args.output / "latest.pt",
        )
    (args.output / "metrics.json").write_text(json.dumps({"config": config, "history": history}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
