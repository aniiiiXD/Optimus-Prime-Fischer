"""Observe the unchanged OneVOne baseline under the official local harness.

The observer runs in the engine process. Agent files, decisions, masking and
clock accounting are unchanged. Late-game masked positions are retained for
offline diagnosis; none of those analyses are supplied to either player.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parent
KIT = ROOT / "official-kit"
sys.path.insert(0, str(KIT))

from catan_kit import match
from catan_kit._patch import ensure_hash_seed
from catan_kit.evaluate import summarise
from catan_kit.obs import masked_copy
from catanatron.game import Game
from catanatron.models.player import RandomPlayer

RESOURCES = ("WOOD", "BRICK", "SHEEP", "WHEAT", "ORE")
DEV_TYPES = ("KNIGHT", "VICTORY_POINT", "ROAD_BUILDING", "YEAR_OF_PLENTY", "MONOPOLY")


def primitive(value):
    if hasattr(value, "name"):
        return value.name
    if isinstance(value, (list, tuple)):
        return [primitive(x) for x in value]
    if isinstance(value, dict):
        return {str(k): primitive(v) for k, v in value.items()}
    return value


def action_json(action):
    return {"type": action.action_type.name, "value": primitive(action.value)}


def player_snapshot(game, color):
    st = game.state
    ps = st.player_state
    prefix = f"P{st.color_to_index[color]}_"
    production = dict.fromkeys(RESOURCES, 0.0)
    effective = dict.fromkeys(RESOURCES, 0.0)
    settlements, cities, roads = [], [], []
    for node, (owner, kind) in st.board.buildings.items():
        if owner != color:
            continue
        (cities if kind == "CITY" else settlements).append(node)
        for tile in st.board.map.adjacent_tiles[node]:
            if tile.resource is None:
                continue
            amount = (6 - abs(7 - tile.number)) / 36 * (2 if kind == "CITY" else 1)
            production[tile.resource] += amount
            if tile is not st.board.map.land_tiles[st.board.robber_coordinate]:
                effective[tile.resource] += amount
    for edge, owner in st.board.roads.items():
        if owner == color:
            roads.append(list(edge))
    return {
        "public_vp": ps[prefix + "VICTORY_POINTS"],
        "actual_vp": ps[prefix + "ACTUAL_VICTORY_POINTS"],
        "resources": {r: ps[prefix + r + "_IN_HAND"] for r in RESOURCES},
        "devs": {d: ps[prefix + d + "_IN_HAND"] for d in DEV_TYPES},
        "played_knights": ps[prefix + "PLAYED_KNIGHT"],
        "has_army": bool(ps[prefix + "HAS_ARMY"]),
        "has_road": bool(ps[prefix + "HAS_ROAD"]),
        "longest_road": ps[prefix + "LONGEST_ROAD_LENGTH"],
        "settlements": sorted(settlements), "cities": sorted(cities),
        "roads": roads,
        "production": {r: round(v, 6) for r, v in production.items()},
        "effective_production": {r: round(v, 6) for r, v in effective.items()},
        "buildable_sites": len(st.board.buildable_node_ids(color)),
    }


class Recorder:
    def __init__(self, args, output):
        self.args = args
        self.output = output
        self.games = []
        self.trace = gzip.open(output / "decisions.jsonl.gz", "wt")
        self.positions = gzip.open(output / "late_positions.pkl.gz", "wb")
        self.position_count = 0

    def start(self, game):
        gi = len(self.games)
        colors = {"a": match.COLORS[gi % 2], "b": match.COLORS[1 - gi % 2]}
        data = {"game": gi, "board_seed": game.seed, "a_first": gi % 2 == 0,
                "checkpoints": {}, "opening": None, "events": [],
                "stats": {s: {"actions": Counter(), "production_drawn": 0,
                              "production_expected": 0.0, "discarded": 0,
                              "trade_cards_spent": 0, "trades": 0,
                              "trade_turns_no_conversion": 0,
                              "first_settlement_turn": None, "first_city_turn": None,
                              "end_turn_high_hand": 0, "own_turns": 0,
                              "end_turn_with_build_available": 0,
                              "end_turn_settlement_affordable_no_site": 0,
                              "blocked_production_expected": 0.0}
                          for s in ("a", "b")}}
        self.games.append(data)
        game.diag = {"data": data, "colors": colors, "step": 0,
                     "turn_actions": {s: Counter() for s in colors}}

    def snapshots(self, game):
        return {s: player_snapshot(game, c) for s, c in game.diag["colors"].items()}

    def before(self, game, action):
        diag, st = game.diag, game.state
        data = diag["data"]
        side = next(s for s, c in diag["colors"].items() if c == action.color)
        pre = self.snapshots(game)
        initial = st.is_initial_build_phase
        if not initial and data["opening"] is None:
            data["opening"] = pre
        for milestone in (10, 20, 40, 60, 80, 100):
            if not initial and st.num_turns >= milestone and str(milestone) not in data["checkpoints"]:
                data["checkpoints"][str(milestone)] = pre
        legal = list(game.playable_actions)
        legal_types = Counter(a.action_type.name for a in legal)
        row = {"game": data["game"], "step": diag["step"], "turn": st.num_turns,
               "side": side, "initial": initial, "prompt": st.current_prompt.name,
               "forced": len(legal) == 1, "action": action_json(action),
               "legal_types": dict(legal_types), "before": pre}
        if side == "a" and not initial and len(legal) > 1 and (
            (pre["a"]["actual_vp"] >= 8 and st.current_prompt.name == "PLAY_TURN")
            or legal_types.get("PLAY_MONOPOLY", 0)
        ):
            world = masked_copy(game, action.color, random.Random(data["game"] * 100000 + diag["step"]))
            world.state.players = [RandomPlayer(c) for c in world.state.colors]
            world.seed = 0
            pickle.dump((row["game"], row["step"], world, action, legal), self.positions,
                        protocol=pickle.HIGHEST_PROTOCOL)
            self.position_count += 1
        return row

    def after(self, game, row, action_record):
        diag, data = game.diag, game.diag["data"]
        post = self.snapshots(game)
        side, kind = row["side"], row["action"]["type"]
        stats = data["stats"][side]
        before, after = row["before"], post
        row["after"] = post
        # Development identities are retained solely as offline trace truth;
        # agents still receive the original official masked observations.
        row["result"] = primitive(action_record.result)
        self.trace.write(json.dumps(row, separators=(",", ":")) + "\n")
        diag["step"] += 1
        if row["initial"]:
            return
        stats["actions"][kind] += 1
        diag["turn_actions"][side][kind] += 1
        if kind == "ROLL":
            stats["own_turns"] += 1
            for s in ("a", "b"):
                data["stats"][s]["production_drawn"] += sum(
                    after[s]["resources"][r] - before[s]["resources"][r] for r in RESOURCES)
                expected = sum(before[s]["effective_production"].values())
                data["stats"][s]["production_expected"] += expected
                data["stats"][s]["blocked_production_expected"] += sum(before[s]["production"].values()) - expected
        elif kind == "DISCARD_RESOURCE":
            stats["discarded"] += 1
        elif kind == "MARITIME_TRADE":
            stats["trades"] += 1
            stats["trade_cards_spent"] += sum(max(0, before[side]["resources"][r] - after[side]["resources"][r]) for r in RESOURCES)
        elif kind == "BUILD_SETTLEMENT" and stats["first_settlement_turn"] is None:
            stats["first_settlement_turn"] = row["turn"]
        elif kind == "BUILD_CITY" and stats["first_city_turn"] is None:
            stats["first_city_turn"] = row["turn"]
        elif kind == "END_TURN":
            turn_actions = diag["turn_actions"][side]
            conversion = sum(turn_actions[t] for t in ("BUILD_SETTLEMENT", "BUILD_CITY", "BUILD_ROAD", "BUY_DEVELOPMENT_CARD"))
            stats["trade_turns_no_conversion"] += bool(turn_actions["MARITIME_TRADE"] and not conversion)
            stats["end_turn_high_hand"] += sum(before[side]["resources"].values()) > 7
            stats["end_turn_with_build_available"] += any(row["legal_types"].get(t) for t in ("BUILD_SETTLEMENT", "BUILD_CITY"))
            hand = before[side]["resources"]
            stats["end_turn_settlement_affordable_no_site"] += bool(
                all(hand[r] >= 1 for r in ("WOOD", "BRICK", "SHEEP", "WHEAT"))
                and not row["legal_types"].get("BUILD_SETTLEMENT"))
            turn_actions.clear()
        if kind in ("BUILD_SETTLEMENT", "BUILD_CITY", "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY", "BUY_DEVELOPMENT_CARD", "MOVE_ROBBER") or after[side]["actual_vp"] != before[side]["actual_vp"]:
            data["events"].append({"step": row["step"], "turn": row["turn"], "side": side,
                                   "action": row["action"], "result": row["result"],
                                   "vp_before": before[side]["actual_vp"],
                                   "vp_after": after[side]["actual_vp"]})

    def finish(self, game):
        d = game.diag["data"]
        d["final"] = self.snapshots(game)
        d["turns"] = game.state.num_turns
        winner = game.winning_color()
        d["winner"] = next((s for s, c in game.diag["colors"].items() if c == winner), None)
        if len(self.games) % 20 == 0:
            print(f"Traced {len(self.games)}/{self.args.games}; A wins {sum(g['winner'] == 'a' for g in self.games)}", flush=True)

    def close(self):
        self.trace.close()
        self.positions.close()


RECORDER = None


class ObservedGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        RECORDER.start(self)

    def execute(self, action, validate_action=True, action_record=None):
        row = RECORDER.before(self, action)
        result = super().execute(action, validate_action, action_record)
        RECORDER.after(self, row, result)
        return result

    def play(self, *args, **kwargs):
        winner = super().play(*args, **kwargs)
        RECORDER.finish(self)
        return winner


def main():
    global RECORDER
    ensure_hash_seed()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed0", type=int, default=2026101800)
    parser.add_argument("--label", default="baseline_trace_200")
    args = parser.parse_args()
    output = ROOT / "reports" / "onevone_analysis" / args.label
    output.mkdir(parents=True, exist_ok=False)
    original = json.loads((ROOT / "reports/onevone_baseline/baseline_200.json").read_text())
    for name, sha in original["agent_files_sha256"].items():
        assert hashlib.sha256((ROOT / "onevone_baseline" / name).read_bytes()).hexdigest() == sha, name
    RECORDER = Recorder(args, output)
    match.Game = ObservedGame
    try:
        records, info = match.play_chunk(str(ROOT / "onevone_baseline/agent.py"),
                                        str(KIT / "examples/f_agent.py"),
                                        [args.seed0 + i // 2 for i in range(args.games)],
                                        list(range(args.games)))
    finally:
        match.Game = Game
        RECORDER.close()
    for trace, record in zip(RECORDER.games, records):
        assert trace["winner"] == record["winner"]
        trace["harness"] = record
    (output / "games.json").write_text(json.dumps(RECORDER.games, indent=2) + "\n")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "games": args.games, "seed0": args.seed0, "opponent": "official F",
               "observation": "Engine-side trace only; agent code and official clock unchanged",
               "a": summarise(records, "a"), "b": summarise(records, "b"),
               "load": info, "late_positions": RECORDER.position_count,
               "original_baseline_wins": original["result"]["wins"],
               "original_baseline_decisions": original["result"]["decisions"]}
    if args.games == original["games"] and args.seed0 == original["seed_start"]:
        summary["matches_original_wins_and_decisions"] = (
            summary["a"]["wins"] == original["result"]["wins"] and
            summary["a"]["decisions"] == original["result"]["decisions"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
