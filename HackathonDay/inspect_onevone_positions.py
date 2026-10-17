"""Offline checks on saved masked positions, never used to choose live moves."""

from __future__ import annotations

import argparse
from collections import Counter, deque
import gzip
import json
from pathlib import Path
import pickle
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "official-kit"))
sys.path.insert(0, str(ROOT / "onevone_baseline"))
from catan_kit._patch import ensure_hash_seed
from catanatron.models.enums import ActionType
from fair_value import FairValueAgent

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
SAFE = {"BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "MARITIME_TRADE",
        "PLAY_YEAR_OF_PLENTY", "PLAY_ROAD_BUILDING", "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY"}
ORDER = {k: i for i, k in enumerate(("BUILD_SETTLEMENT", "BUILD_CITY", "PLAY_KNIGHT_CARD",
                                  "BUILD_ROAD", "PLAY_MONOPOLY", "PLAY_YEAR_OF_PLENTY",
                                  "MARITIME_TRADE", "PLAY_ROAD_BUILDING"))}


def action_json(action):
    return {"type": action.action_type.name, "value": action.value}


def own_vp(game, color):
    return game.state.player_state[f"P{game.state.color_to_index[color]}_ACTUAL_VICTORY_POINTS"]


def state_key(game):
    st = game.state
    return (tuple(sorted(st.player_state.items())), tuple(sorted(st.board.buildings.items())),
            tuple(sorted(st.board.roads.items())), tuple(st.resource_freqdeck),
            st.current_player_index, st.current_prompt,
            tuple((a.action_type.name, repr(a.value)) for a in game.playable_actions))


def winning_line(game, color, max_depth=3, max_nodes=1500):
    queue = deque([(game, [])])
    seen = {state_key(game)}
    nodes = 0
    while queue:
        world, line = queue.popleft()
        if len(line) >= max_depth or world.state.current_color() != color:
            continue
        moves = sorted((a for a in world.playable_actions if a.action_type.name in SAFE),
                       key=lambda a: (ORDER[a.action_type.name], repr(a.value)))
        for action in moves:
            if nodes >= max_nodes:
                return None, nodes, True
            after = world.copy()
            after.execute(action)
            nodes += 1
            next_line = line + [action_json(action)]
            if own_vp(after, color) >= after.vps_to_win:
                return next_line, nodes, False
            if len(next_line) < max_depth:
                key = state_key(after)
                if key not in seen:
                    seen.add(key)
                    queue.append((after, next_line))
    return None, nodes, False


def main():
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="baseline_trace_200")
    args = parser.parse_args()
    folder = ROOT / "reports/onevone_analysis" / args.label
    outcomes = {g["game"]: g["winner"] for g in json.loads((folder / "games.json").read_text())}
    cfg = json.loads((ROOT / "onevone_baseline/config.json").read_text())
    counts = Counter()
    missed = []
    mono = []
    dev = []
    searches = []
    with gzip.open(folder / "late_positions.pkl.gz", "rb") as stream:
        while True:
            try:
                gi, step, world, chosen, legal = pickle.load(stream)
            except EOFError:
                break
            counts["positions"] += 1
            color = chosen.color
            st = world.state
            own = f"P{st.color_to_index[color]}_"
            opp_color = next(c for c in st.colors if c != color)
            opp = f"P{st.color_to_index[opp_color]}_"
            vp = own_vp(world, color)
            resources = {r: st.player_state[own + r + "_IN_HAND"] for r in RESOURCES}
            opp_resources = {r: st.player_state[opp + r + "_IN_HAND"] for r in RESOURCES}
            kind = chosen.action_type.name
            if vp >= 8:
                counts["positions_at_8plus"] += 1
                counts[f"choice_8plus_{kind}"] += 1
            if kind == "PLAY_MONOPOLY":
                scorer = FairValueAgent(color, cfg)
                candidates = []
                production = scorer.robber._production(world, opp_color)
                for action in legal:
                    if action.action_type != ActionType.PLAY_MONOPOLY:
                        continue
                    after = world.copy()
                    after.execute(action)
                    candidates.append({"resource": action.value,
                                       "actual_cards": opp_resources[action.value],
                                       "estimated_cards": min(2.0, 0.25 + 6.0 * production[action.value]),
                                       "exact_successor_score": scorer.value_fn(after, color)})
                best = max(candidates, key=lambda c: c["exact_successor_score"])
                selected = next(c for c in candidates if c["resource"] == chosen.value)
                row = {"game": gi, "step": step, "turn": st.num_turns,
                       "outcome": outcomes[gi], "vp": vp, "resources": resources,
                       "opponent_resources": opp_resources, "selected": selected,
                       "best_exact_resource": best,
                       "strict_exact_score_regret": best["exact_successor_score"] > selected["exact_successor_score"],
                       "candidates": candidates}
                mono.append(row)
            suspicious_buy = kind == "BUY_DEVELOPMENT_CARD" and vp == 9
            if suspicious_buy:
                counts["nine_vp_buy_scan_skipped"] += 1
                dev.append({"game": gi, "step": step, "turn": st.num_turns, "outcome": outcomes[gi]})
            if vp >= 8 and (kind == "END_TURN" or suspicious_buy):
                line, nodes, truncated = winning_line(world, color)
                row = {"game": gi, "step": step, "turn": st.num_turns, "outcome": outcomes[gi],
                       "vp": vp, "chosen": action_json(chosen), "resources": resources,
                       "opponent_public_vp": st.player_state[opp + "VICTORY_POINTS"],
                       "winning_line": line, "nodes": nodes, "truncated": truncated}
                searches.append(row)
                if line:
                    missed.append(row)
    result = {"counts": dict(counts),
              "method": "Offline deterministic search at own VP>=8 when ending a turn or buying development at 9 VP; depth<=3, 1500 expansions per position; masked states only. Absence of a found line is not proof no deeper win exists.",
              "searched_positions": len(searches),
              "searches_truncated": sum(x["truncated"] for x in searches),
              "missed_winning_lines": missed,
              "monopoly_plays": mono,
              "nine_vp_development_purchases": dev,
              "searches": searches}
    (folder / "position_checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"counts": dict(counts), "searched_positions": len(searches),
                      "searches_truncated": result["searches_truncated"],
                      "found_winning_alternatives": len(missed),
                      "monopoly_plays": len(mono),
                      "monopoly_zero_cards": sum(x["selected"]["actual_cards"] == 0 for x in mono),
                      "monopoly_strict_exact_score_regret": sum(x["strict_exact_score_regret"] for x in mono),
                      "missed_examples": missed[:6], "monopoly_examples": mono[:3]}, indent=2))


if __name__ == "__main__":
    main()
