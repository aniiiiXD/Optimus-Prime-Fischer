"""Audit a Fireworks LLM as a GPU-backed Connect Four move-ranking teacher.

The LLM does not supply trusted targets. Its ranking is validated against the
rules engine and stored alongside local alpha-beta labels so we can measure its
usefulness before any training decision is made.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect_four_engine import ConnectFour
from connect_four_transformer.generate_dataset import (
    Candidate,
    collect_candidates,
    label_candidate,
)
from connect_four_transformer.search import AlphaBetaTeacher


DEFAULT_MODEL = "accounts/fireworks/models/ember-1"
DEFAULT_OUTPUT = Path(__file__).parent / "artifacts" / "data" / "fireworks_teacher_audit.jsonl"
CHAT_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


def board_text(game: ConnectFour) -> str:
    symbols = {1: "X", -1: "O", 0: "."}
    rows = [" ".join(symbols[game.board[row * 7 + column]] for column in range(7)) for row in range(6)]
    return "\n".join(["1 2 3 4 5 6 7", *rows])


def prompt_for(game: ConnectFour) -> str:
    player = "X" if game.current_player == 1 else "O"
    legal_human = [action + 1 for action in game.legal_actions()]
    return f"""You are ranking legal moves for standard Connect Four. Four in a row wins.
You are {player} to move. Board rows are top to bottom; '.' is empty.

{board_text(game)}

Legal columns are exactly: {legal_human}.
Return JSON only, with this exact shape:
{{"ranking":[best_column,second_column,third_column],"reason":"at most 18 words"}}
The ranking must contain 1 to 3 distinct legal columns and no other text."""


def request_completion(api_key: str, model: str, prompt: str, max_tokens: int) -> tuple[str, dict[str, Any], float]:
    """Force the move ranking through a tool call instead of trusting prose.

    Some reasoning models spend most of a small completion budget thinking and
    either omit ``content`` or narrate their analysis despite JSON mode. Tools
    give the API a typed final-answer channel; the local parser still validates
    columns against the actual board.
    """
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": "Solve the position, then call the required tool exactly once. Never invent an illegal move."},
            {"role": "user", "content": prompt},
        ],
        "tools": [{
            "type": "function",
            "function": {
                "name": "submit_move_ranking",
                "description": "Submit one to three ranked legal Connect Four columns.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ranking": {
                            "type": "array", "minItems": 1, "maxItems": 3,
                            "items": {"type": "integer", "minimum": 1, "maximum": 7},
                        },
                        "reason": {"type": "string", "maxLength": 180},
                    },
                    "required": ["ranking", "reason"],
                    "additionalProperties": False,
                },
            },
        }],
        "tool_choice": {"type": "function", "function": {"name": "submit_move_ranking"}},
    }
    request = Request(
        CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urlopen(request, timeout=60) as response:
            body = json.load(response)
    except HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Fireworks HTTP {error.code}: {detail}") from error
    except URLError as error:
        raise RuntimeError(f"Fireworks network error: {error.reason}") from error
    elapsed = time.perf_counter() - started
    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"unexpected Fireworks response shape: {body}") from error
    tool_calls = message.get("tool_calls") or []
    if tool_calls:
        try:
            text = tool_calls[0]["function"]["arguments"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError(f"unexpected Fireworks tool-call shape: {body}") from error
    else:
        text = message.get("content")
    return str(text), body, elapsed


def parse_ranking(text: str, legal_actions: tuple[int, ...]) -> tuple[list[int] | None, str | None]:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*|\s*```$", "", clean).strip()
    try:
        response = json.loads(clean)
    except json.JSONDecodeError:
        return None, "response_is_not_json"
    ranking = response.get("ranking") if isinstance(response, dict) else None
    if not isinstance(ranking, list) or not 1 <= len(ranking) <= 3:
        return None, "ranking_must_have_one_to_three_columns"
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ranking):
        return None, "ranking_columns_must_be_integers"
    actions = [column - 1 for column in ranking]
    if len(set(actions)) != len(actions):
        return None, "ranking_contains_duplicates"
    if any(action not in legal_actions for action in actions):
        return None, "ranking_contains_illegal_column"
    return actions, None


def ranking_policy(actions: list[int] | None) -> list[float] | None:
    if not actions:
        return None
    weights = (0.64, 0.26, 0.10)
    policy = [0.0] * 7
    used = weights[:len(actions)]
    normalizer = sum(used)
    for action, weight in zip(actions, used):
        policy[action] = weight / normalizer
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2051)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--node-limit", type=int, default=5_000)
    parser.add_argument("--temperature", type=float, default=0.18, help="local alpha-beta soft-policy temperature")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--min-legal-actions", type=int, default=3,
        help="skip near-full trivial boards; collect candidates until this many columns are playable",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if (args.positions <= 0 or args.depth <= 0 or args.node_limit <= 0
            or args.max_tokens <= 0 or not 1 <= args.min_legal_actions <= 7):
        parser.error("positions, depth, node-limit, max-tokens, and min-legal-actions must be positive and valid")
    api_key = os.environ.get("FIREWORKS_API_KEY")
    if not api_key:
        parser.error("FIREWORKS_API_KEY is not set; run: set -a; source .env; set +a")

    # The mixed generator includes late-game states by design. That is useful
    # for a normal corpus, but a GPU-LLM audit should not burn requests on a
    # forced single-column move. Oversample locally, then retain nontrivial
    # legal-choice states only.
    pool = collect_candidates(args.positions * 8, args.seed)
    candidates = [
        candidate for candidate in pool
        if len(ConnectFour.from_moves(candidate.moves).legal_actions()) >= args.min_legal_actions
    ][:args.positions]
    if len(candidates) < args.positions:
        raise RuntimeError(
            f"only found {len(candidates)} states with at least {args.min_legal_actions} legal actions"
        )
    teacher = AlphaBetaTeacher()
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        game = ConnectFour.from_moves(candidate.moves)
        local = label_candidate(
            candidate, teacher, depth=args.depth, node_limit=args.node_limit, temperature=args.temperature
        )
        prompt = prompt_for(game)
        raw, response, seconds = request_completion(api_key, args.model, prompt, args.max_tokens)
        ranking, rejection_reason = parse_ranking(raw, game.legal_actions())
        teacher_best = [action for action in game.legal_actions() if local["policy_target"][action] > 0]
        usage = response.get("usage", {}) if isinstance(response, dict) else {}
        rows.append({
            **local,
            "llm_teacher": {
                "provider": "fireworks",
                "model": args.model,
                "prompt": prompt,
                "raw_response": raw,
                "ranked_actions": ranking,
                "ranking_policy": ranking_policy(ranking),
                "valid": ranking is not None,
                "rejection_reason": rejection_reason,
                "top1_matches_search": bool(ranking and ranking[0] in teacher_best),
                "top3_contains_search_best": bool(ranking and any(action in teacher_best for action in ranking)),
                "latency_seconds": seconds,
                "usage": usage,
            },
        })
        status = "valid" if ranking is not None else rejection_reason
        print(f"{index:>3}/{len(candidates)} {status:>35} | {seconds * 1000:7.1f} ms", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, separators=(",", ":")) + "\n")

    valid = [row for row in rows if row["llm_teacher"]["valid"]]
    latencies = [float(row["llm_teacher"]["latency_seconds"]) for row in rows]
    top1 = sum(row["llm_teacher"]["top1_matches_search"] for row in valid)
    top3 = sum(row["llm_teacher"]["top3_contains_search_best"] for row in valid)
    total_tokens = sum(int(row["llm_teacher"]["usage"].get("total_tokens", 0)) for row in rows)
    print(f"saved {len(rows)} audited positions to {args.output}")
    print(f"legal JSON rankings: {len(valid)}/{len(rows)} | top-1 search agreement: {top1}/{len(valid) if valid else 0} | top-3 contains search best: {top3}/{len(valid) if valid else 0}")
    print(f"API latency median: {statistics.median(latencies) * 1000:.1f} ms | reported total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
