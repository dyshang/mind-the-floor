"""Label-aware constant floor on WebArena (Section 5, Appendix C.5).

One constant answer joins every must_include reference of the tasks whose only
check is string_match with must_include references, and stops at once. Scoring
ports WebArena's own StringEvaluator (clean_answer / exact_match /
must_include, nltk word_tokenize for one-character references) and
EvaluatorComb (product over eval_types). Conservative: url_match,
program_html and anything sent to the LLM judge count as failures, so the
result is a lower bound on this constant's score.

Needs nltk with the punkt_tab tokenizer data. Writes
results/webarena_label_aware.json; scripts/check_appendix_numbers.py checks it
offline.

Run: python scripts/label_aware_webarena.py --webarena-config PATH/test.raw.json
     (web-arena-x/webarena, config_files/test.raw.json)
"""
import argparse
import json
import re
from pathlib import Path

from nltk.tokenize import word_tokenize

REPO = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--webarena-config", required=True)
args = ap.parse_args()
tasks = json.load(open(args.webarena_config))
recorded = json.load(open(REPO / "results" / "webarena_unachievable.json"))


def clean_answer(answer: str) -> str:  # verbatim logic of StringEvaluator.clean_answer
    answer = answer.strip()
    if answer.startswith("'") and answer.endswith("'"):
        answer = answer[1:-1]
    elif answer.startswith('"') and answer.endswith('"'):
        answer = answer[1:-1]
    return answer.lower()


def exact_match(ref, pred):
    return float(clean_answer(pred) == clean_answer(ref))


def must_include(ref, pred, tokenize=False):
    clean_ref, clean_pred = clean_answer(ref), clean_answer(pred)
    if tokenize and len(clean_ref) == 1 and len(word_tokenize(clean_ref)) == 1:
        return float(clean_ref in word_tokenize(clean_pred))
    return float(clean_ref in clean_pred)


def string_score(cfg, pred):  # StringEvaluator.__call__, LLM branches scored 0
    pred = clean_answer(pred)
    score = 1.0
    for approach, value in cfg["eval"]["reference_answers"].items():
        if approach == "exact_match":
            score *= exact_match(value, pred)
        elif approach == "must_include":
            for v in value:
                score *= must_include(v, pred, tokenize=(len(value) == 1))
        elif approach == "fuzzy_match":
            if value == "N/A":
                score *= exact_match(value, pred)
                if score != 1:
                    score = 0.0  # would go to the LLM ua_match judge: counted as a fail
            else:
                score = 0.0      # LLM fuzzy judge: counted as a fail
    return score


def task_score(cfg, pred):  # EvaluatorComb: product over eval_types
    score = 1.0
    for et in cfg["eval"]["eval_types"]:
        score *= string_score(cfg, pred) if et == "string_match" else 0.0
    return score


assert len(tasks) == recorded["n_tasks"] == 812
na_ids = sorted(t["task_id"] for t in tasks if t["eval"]["reference_answers"] and
                t["eval"]["reference_answers"].get("fuzzy_match") == "N/A")
assert na_ids == sorted(recorded["task_ids"]), "config differs from the one the paper counted"

pool = [t for t in tasks if t["eval"]["eval_types"] == ["string_match"]
        and set((t["eval"]["reference_answers"] or {}).keys()) == {"must_include"}]
refs = [v for t in pool for v in t["eval"]["reference_answers"]["must_include"]]
answer = " | ".join(refs)
# The agent's action is parsed by browser_env/actions.py: re.search(r"stop ?\[(.+)\]", s).
# It is greedy within one line, so inner brackets survive and only a newline would not.
_m = re.search(r"stop ?\[(.+)\]", f"stop [{answer}]")
assert "\n" not in answer and _m and _m.group(1) == answer, "answer would not survive the stop [..] parser"

passed = [t["task_id"] for t in tasks if task_score(t, answer) == 1.0]
na_passed = [t["task_id"] for t in tasks if task_score(t, "N/A") == 1.0]
res = {
    "n_tasks": len(tasks),
    "string_match_only_must_include_tasks": len(pool),
    "must_include_references_joined": len(refs),
    "must_include_references_distinct": len(set(refs)),
    "answer_chars": len(answer),
    "constant_answer_passes": len(passed),
    "constant_answer_score": round(len(passed) / len(tasks), 4),
    "constant_NA_passes (paper's policy)": len(na_passed),
    "constant_NA_score": round(len(na_passed) / len(tasks), 4),
    "pool_tasks_failed": sorted(set(t["task_id"] for t in pool) - set(passed)),
}
print(json.dumps(res, indent=1))
res["source"] = "web-arena-x/webarena, config_files/test.raw.json (same task set as webarena_unachievable.json: its 36 N/A ids match)"
res["generator"] = "scripts/label_aware_webarena.py"
(REPO / "results" / "webarena_label_aware.json").write_text(json.dumps({**res, "passed_task_ids": passed}, indent=1) + "\n")
