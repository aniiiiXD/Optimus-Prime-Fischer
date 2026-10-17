"""Compare matched board pairs without treating two seats as independent boards."""
import argparse
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def paired(control, candidate):
    a, b = control["records"], candidate["records"]
    if len(a) != len(b) or len(a)%2:
        raise ValueError("Need equal complete seat pairs")
    if any((x["board_seed"], x["a_first"]) != (y["board_seed"], y["a_first"]) for x,y in zip(a,b)):
        raise ValueError("Board/seat schedules differ")
    differences = [int(y["winner"] == "a")-int(x["winner"] == "a") for x,y in zip(a,b)]
    pairs = [sum(differences[i:i+2])/2 for i in range(0,len(a),2)]
    rng = random.Random(1729)
    bootstrap = sorted(100*sum(rng.choices(pairs,k=len(pairs)))/len(pairs) for _ in range(10000))
    return {"difference_pp": 100*sum(differences)/len(a),
            "board_bootstrap_95_pp": [bootstrap[250], bootstrap[9749]],
            "gained_games": differences.count(1), "lost_games": differences.count(-1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="final_v2")
    args = parser.parse_args()
    root = ROOT/"reports"
    reports = {name: json.loads((root/f"{args.prefix}_{name}.json").read_text())
               for name in ("learned_f_200", "original_f_200", "heuristic_f_200", "head_to_head_200")}
    if any(r["status"] != "completed" or not r["source_unchanged"] or not r["core_unchanged"] for r in reports.values()):
        raise AssertionError("A run failed or its source changed")
    selection = json.loads((root/"selection_v2.json").read_text())
    checkpoint = ROOT/selection["checkpoint"]
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection["checkpoint_sha256"]:
        raise AssertionError("Selected checkpoint changed")
    training = json.loads((checkpoint.parent/"training.json").read_text())
    seen = set(training["train_seeds"]+training["validation_seeds"])
    final = {r["board_seed"] for report in reports.values() for r in report["records"]}
    if seen & final:
        raise AssertionError("Training/test board leakage")
    fields = ("wins", "games", "win_rate", "mean_cpu_s", "worst_cpu_s", "games_over_budget",
              "random_moves", "errors", "timeouts", "crashes")
    result = {"runs": {name:{k:r["a"][k] for k in fields} for name,r in reports.items()},
              "learned_minus_original": paired(reports["original_f_200"], reports["learned_f_200"]),
              "learned_minus_same_heuristic": paired(reports["heuristic_f_200"], reports["learned_f_200"]),
              "heuristic_minus_original": paired(reports["original_f_200"], reports["heuristic_f_200"]),
              "whole_board_split_disjoint": True, "selected_checkpoint_unchanged": True,
              "all_run_source_freezes_passed": True}
    output = root/f"{args.prefix}_comparison.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
