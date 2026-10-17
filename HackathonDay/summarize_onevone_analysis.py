"""Summarize recorded games and position checks without rerunning agents."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
FOLDER = ROOT / "reports/onevone_analysis/baseline_trace_200"


def summarize(games):
    result = {"games": len(games), "mean_turns": mean(g["turns"] for g in games)}
    for side in ("a", "b"):
        stats = [g["stats"][side] for g in games]
        final = [g["final"][side] for g in games]
        fields = ("production_drawn", "production_expected", "blocked_production_expected",
                  "discarded", "trades", "trade_cards_spent", "trade_turns_no_conversion",
                  "end_turn_high_hand", "end_turn_with_build_available",
                  "end_turn_settlement_affordable_no_site")
        data = {f"mean_{k}": mean(s[k] for s in stats) for k in fields}
        for key in ("first_settlement_turn", "first_city_turn"):
            values = [s[key] for s in stats if s[key] is not None]
            data[key] = {"mean_when_built": mean(values) if values else None,
                         "games_built": len(values), "games_never_built": len(games) - len(values)}
        data["opening_mean_production"] = mean(sum(g["opening"][side]["production"].values()) for g in games)
        data["final_mean_vp"] = mean(s["actual_vp"] for s in final)
        data["final_vp_histogram"] = dict(sorted(Counter(s["actual_vp"] for s in final).items()))
        data["final_longest_road_holder_games"] = sum(s["has_road"] for s in final)
        data["final_largest_army_holder_games"] = sum(s["has_army"] for s in final)
        data["final_mean_cities"] = mean(len(s["cities"]) for s in final)
        data["mean_actions"] = {k: mean(s["actions"].get(k, 0) for s in stats)
                                for k in sorted({k for s in stats for k in s["actions"]})}
        checkpoints = {}
        for turn in (10, 20, 40, 60, 80, 100):
            rows = [g["checkpoints"][str(turn)][side] for g in games if str(turn) in g["checkpoints"]]
            if rows:
                checkpoints[str(turn)] = {
                    "games_still_running": len(rows), "mean_vp": mean(r["actual_vp"] for r in rows),
                    "mean_production": mean(sum(r["production"].values()) for r in rows)}
        data["checkpoints"] = checkpoints
        result[side] = data
    result["winning_actions"] = dict(Counter(g["events"][-1]["action"]["type"] for g in games))
    return result


def main():
    games = json.loads((FOLDER / "games.json").read_text())
    checks = json.loads((FOLDER / "position_checks.json").read_text())
    alternatives = checks["missed_winning_lines"]
    mono = checks["monopoly_plays"]
    result = {
        "definitions": {
            "a": "Frozen OneVOne baseline", "b": "Official F agent",
            "turn": "Engine turn counter spanning both players; not a player's own turn count",
            "production": "Expected cards per dice roll, before robber blocking and bank shortages",
            "first_build": "Mean among games where built; never-built games reported separately",
            "cohorts": "Grouped by final outcome; descriptive associations, not causal estimates"},
        "wins": summarize([g for g in games if g["winner"] == "a"]),
        "losses": summarize([g for g in games if g["winner"] == "b"]),
        "seat_wins": {"first": sum(g["winner"] == "a" for g in games if g["a_first"]),
                      "second": sum(g["winner"] == "a" for g in games if not g["a_first"])},
        "board_pair_outcomes": dict(Counter(games[i]["winner"] + games[i+1]["winner"] for i in range(0, len(games), 2))),
        "position_audit": {
            "saved_positions": checks["counts"]["positions"],
            "searched_positions": checks["searched_positions"],
            "truncated_searches": checks["searches_truncated"],
            "guaranteed_alternative_positions": len(alternatives),
            "guaranteed_alternative_games": len({r["game"] for r in alternatives}),
            "guaranteed_alternatives_in_losses": sum(r["outcome"] == "b" for r in alternatives),
            "alternative_choice_types": dict(Counter(r["chosen"]["type"] for r in alternatives)),
            "monopoly_plays": len(mono),
            "zero_card_monopolies": sum(r["selected"]["actual_cards"] == 0 for r in mono),
            "zero_card_monopolies_in_losses": sum(r["selected"]["actual_cards"] == 0 and r["outcome"] == "b" for r in mono),
            "inferior_exact_score_monopolies": sum(r["strict_exact_score_regret"] for r in mono),
            "inferior_exact_score_monopolies_in_losses": sum(r["strict_exact_score_regret"] and r["outcome"] == "b" for r in mono)},
    }
    manifest = json.loads((ROOT / "reports/onevone_baseline/baseline_200.json").read_text())
    result["baseline_hashes_match_original"] = all(
        hashlib.sha256((ROOT / "onevone_baseline" / name).read_bytes()).hexdigest() == digest
        for name, digest in manifest["agent_files_sha256"].items())
    assert result["baseline_hashes_match_original"]
    (FOLDER / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"wins": result["wins"]["games"], "losses": result["losses"]["games"],
                      "position_audit": result["position_audit"],
                      "baseline_hashes_match_original": result["baseline_hashes_match_original"]}, indent=2))


if __name__ == "__main__":
    main()
