"""Score BFCL degenerate predictions with the official AST checker.

Thin wrapper: loads the constant-policy prediction JSONL written by
`bfcl_degenerate.py --generate`, feeds each already-decoded call to
`bfcl_eval.eval_checker.ast_eval.ast_checker.ast_checker` (unmodified,
from the official gorilla repo), and reports AST accuracy.

Run with a venv that has `bfcl_eval` installed (py3.11; numpy pin breaks 3.13):
  venv-bfcl/bin/python scripts/executed/bfcl_score.py \
      --data-dir <gorilla>/berkeley-function-call-leaderboard/bfcl_eval/data \
      --category simple_python live_simple live_multiple

Also scores the structural irrelevance floor: a constant no-call policy is
correct on every irrelevance entry by definition of the metric.
Output: results/bfcl_scores.json + per-entry results/bfcl_eval_<cat>.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
# Any model registered with underscore_to_dot=False keeps dotted function
# names intact; the checker only uses the name for that lookup.
MODEL_NAME = "gorilla-openfunctions-v2"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def score_category(data_dir: Path, category: str) -> dict:
    from bfcl_eval.constants.enums import Language
    from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker

    tests = {e["id"]: e for e in load_jsonl(data_dir / f"BFCL_v4_{category}.json")}
    answers = {e["id"]: e for e in
               load_jsonl(data_dir / "possible_answer" / f"BFCL_v4_{category}.json")}
    preds = load_jsonl(RESULTS / f"bfcl_degenerate_BFCL_v4_{category}.jsonl")

    per_entry = []
    correct = 0
    for pred in preds:
        entry = tests[pred["id"]]
        gt = answers[pred["id"]]["ground_truth"]
        result = ast_checker(
            entry["function"],
            pred["result"],
            gt,
            Language.PYTHON,
            category,
            MODEL_NAME,
        )
        correct += bool(result["valid"])
        per_entry.append({"id": pred["id"], "valid": result["valid"],
                          "error_type": result.get("error_type", "")})

    out = RESULTS / f"bfcl_eval_{category}.jsonl"
    with open(out, "w") as f:
        for row in per_entry:
            f.write(json.dumps(row) + "\n")
    n = len(preds)
    return {"category": category, "n": n, "correct": correct,
            "ast_accuracy": correct / n, "policy": "constant-modal-function",
            "checker": "official bfcl_eval ast_checker",
            # Repo-relative: an absolute path would leak the author's home
            # directory into a released artifact, and this is the form the
            # paper uses when it cites these files.
            "per_entry": f"results/{out.name}"}


def irrelevance_floor(data_dir: Path, category: str) -> dict:
    """Constant silence is correct on every irrelevance entry: the metric
    scores abstention, and the degenerate policy always abstains."""
    n = len(load_jsonl(data_dir / f"BFCL_v4_{category}.json"))
    return {"category": category, "n": n, "correct": n, "ast_accuracy": 1.0,
            "policy": "constant-no-call (structural)",
            "checker": "structural: metric counts abstention as correct"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--category", nargs="+",
                        default=["simple_python", "live_simple", "live_multiple"])
    args = parser.parse_args()

    scores = [score_category(args.data_dir, c) for c in args.category]
    scores.append(irrelevance_floor(args.data_dir, "irrelevance"))
    scores.append(irrelevance_floor(args.data_dir, "live_irrelevance"))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "bfcl_scores.json"
    out.write_text(json.dumps(scores, indent=2))
    for s in scores:
        print(f"{s['category']}: {s['correct']}/{s['n']} = {s['ast_accuracy']:.4f}"
              f"  [{s['policy']}]")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
