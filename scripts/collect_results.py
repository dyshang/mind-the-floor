"""Merge raw audit outputs + SOTA references into figure/table-ready JSON.

Inputs (results/): analytic_floors.json (from analytic_floors.py),
bfcl_scores.json (from executed/bfcl_score.py), alfworld_*.jsonl,
tau_*_*.jsonl (optional), sota.json (hand-entered, dated).
Outputs: results/executed_floors.json and sota-annotated
analytic_floors.json in place; both feed make_figure1.py.

Run: .venv/bin/python scripts/collect_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

# analytic_floors.json benchmark name -> sota.json key
ANALYTIC_SOTA_KEY = {
    "MMLU-Pro": "MMLU-Pro",
    "GPQA (diamond)": "GPQA-diamond",
    "GSM8K": "GSM8K",
    "SWE-bench Lite": "SWE-bench Lite",
    "GAIA": "GAIA",
    "WebArena": "WebArena",
}


def jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    sota = json.loads((RESULTS / "sota.json").read_text())

    # --- annotate analytic entries with SOTA ---
    analytic = json.loads((RESULTS / "analytic_floors.json").read_text())
    for entry in analytic:
        key = ANALYTIC_SOTA_KEY.get(entry.get("benchmark"))
        if key and key in sota:
            entry["sota"] = sota[key]["sota"]
            entry["sota_model"] = sota[key].get("model")
            entry["sota_source"] = sota[key]["source"]
            entry["sota_as_of"] = sota[key].get("as_of")
            entry["sota_verified"] = sota[key]["verified"]
    (RESULTS / "analytic_floors.json").write_text(json.dumps(analytic, indent=2))

    # --- executed group ---
    executed: list[dict] = []

    bfcl_path = RESULTS / "bfcl_scores.json"
    if bfcl_path.exists():
        for row in json.loads(bfcl_path.read_text()):
            key = f"BFCL_v4_{row['category']}"
            metric = ("irrelevance_accuracy" if "irrelevance" in row["category"]
                      else "ast_accuracy")
            executed.append({
                "benchmark": f"BFCL ({row['category'].replace('_', ' ')})",
                "split": f"BFCL_v4_{row['category']}",
                "metric": metric,
                "n": row["n"],
                "floor": row["ast_accuracy"],
                "policy": row["policy"],
                "provenance": row["checker"],
                "sota": sota.get(key, sota.get("BFCL_v4_irrelevance", {})).get("sota")
                        if "irrelevance" in row["category"]
                        else sota.get(key, {}).get("sota"),
            })

    for policy in ("fixed", "random"):
        path = RESULTS / f"alfworld_{policy}.jsonl"
        if path.exists():
            rows = jsonl(path)
            n = len(rows)
            if n:
                executed.append({
                    "benchmark": f"ALFWorld ({policy})",
                    "split": "eval_out_of_distribution",
                    "metric": "success_rate",
                    "n": n,
                    "floor": sum(r["success"] for r in rows) / n,
                    "policy": f"{policy} legal action, seed 0",
                    "provenance": "executed via official AlfredTWEnv",
                    "sota": sota.get("ALFWorld", {}).get("sota"),
                })

    for env in ("retail", "airline"):
        for policy in ("transfer", "stop"):
            path = RESULTS / f"tau_{env}_{policy}.jsonl"
            if path.exists():
                rows = jsonl(path)
                n = len(rows)
                if n:
                    executed.append({
                        "benchmark": f"tau-bench {env} ({policy})",
                        "split": "test",
                        "metric": "pass_hat_1",
                        "n": n,
                        "floor": sum(r["reward"] for r in rows) / n,
                        "policy": f"constant {policy}, seed 0",
                        "provenance": "executed via official tau-bench env",
                        "sota": sota.get(f"tau-bench-{env}", {}).get("sota"),
                    })

    out = RESULTS / "executed_floors.json"
    out.write_text(json.dumps(executed, indent=2))
    print(f"wrote {out} ({len(executed)} entries); annotated analytic_floors.json")
    for e in executed:
        cr = f"CR={e['floor']/e['sota']:.3f}" if e.get("sota") else "CR=?"
        print(f"  {e['benchmark']:<28} floor={e['floor']:.4f}  {cr}")


if __name__ == "__main__":
    main()
