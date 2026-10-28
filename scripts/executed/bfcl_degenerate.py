"""BFCL degenerate policy: constant most-frequent function + default args.

Two stages:
  1. `--stats`: scan BFCL test data, report the function-name distribution and
     the modal function per category (this alone gives the analytic part).
  2. `--generate`: write a BFCL-format result file in which every entry calls
     the modal function with schema-default (or type-default) arguments, ready
     for the official checker:
       git clone https://github.com/ShishirPatil/gorilla && cd berkeley-function-call-leaderboard
       pip install -e . && bfcl evaluate --model degenerate-constant ...

Scoring runs through the official AST checker only; we never modify it.
Output: results/bfcl_degenerate.jsonl (one entry per test case).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "results"
SEED = 0  # no randomness in the constant policy; recorded for the protocol


def type_default(param_schema: dict):
    t = param_schema.get("type", "string")
    return {"string": "", "integer": 0, "number": 0.0, "boolean": False,
            "array": [], "object": {}, "dict": {}, "float": 0.0}.get(t, "")


def modal_function(test_file: Path) -> tuple[str, dict, Counter]:
    """Most frequent function name across the category's ground truth."""
    names: Counter = Counter()
    schemas: dict = {}
    with open(test_file) as f:
        for line in f:
            entry = json.loads(line)
            for fn in entry.get("function", []):
                names[fn["name"]] += 1
                schemas.setdefault(fn["name"], fn)
    if not names:
        raise SystemExit(f"no function schemas found in {test_file}")
    mode, _ = names.most_common(1)[0]
    return mode, schemas[mode], names


def constant_call(schema: dict) -> dict:
    params = schema.get("parameters", {})
    required = params.get("required", [])
    props = params.get("properties", {})
    args = {name: props[name].get("default", type_default(props[name]))
            for name in required if name in props}
    return {schema["name"]: args}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("test_file", type=Path,
                        help="BFCL category test JSONL (e.g. BFCL_v3_simple.json)")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--generate", action="store_true")
    args = parser.parse_args()

    mode, schema, names = modal_function(args.test_file)
    total = sum(names.values())
    print(f"functions: {len(names)}; modal: {mode} "
          f"({names[mode]}/{total} = {names[mode]/total:.4f})")

    if args.generate:
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"bfcl_degenerate_{args.test_file.stem}.jsonl"
        call = constant_call(schema)
        with open(args.test_file) as f, open(out, "w") as g:
            for line in f:
                entry = json.loads(line)
                g.write(json.dumps({
                    "id": entry["id"],
                    "result": [call],
                    "policy": "constant-modal-function",
                    "seed": SEED,
                }) + "\n")
        print(f"wrote {out}; score it with the official BFCL checker")


if __name__ == "__main__":
    main()
