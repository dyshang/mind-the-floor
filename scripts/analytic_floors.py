"""Compute analytic-group floors from public label statistics.

Downloads label columns only (via HF datasets), computes closed-form floors
with floorcheck, and updates results/analytic_floors.json. The positive
control (SWE-bench Lite) is a structural lower bound; GAIA is counted from its
answer file (see `gaia`); WebArena is a count over released task configs (see
`webarena`).

Run: .venv/bin/python scripts/analytic_floors.py [--skip-download]
                                                 [--webarena-config PATH]

This file is a MERGE TARGET, not a fresh artifact. No single invocation can
produce all of it -- downloads may be skipped, GPQA is gated, and WebArena's
configs live upstream -- so each run updates the entries it can compute and
leaves the rest alone. Two consequences worth knowing:

  * The SOTA fields are owned by collect_results.py, which merges them in from
    sota.json. This script never fills them, and never overwrites them either.
    Run collect_results.py after this one and before make_figure1.py.
  * If an expected entry is missing the script refuses to write, rather than
    emitting a short file that looks complete. --allow-partial overrides.

GPQA is gated on HF (accept terms + `huggingface-cli login` first); if it is
unavailable the run reports the failure and keeps the recorded entry.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from floorcheck import accuracy_floor, exact_match_floor, pass_at_k_floor

RESULTS = Path(__file__).resolve().parent.parent / "results"

# SOTA figures are snapshots; verify and date them at paper-writing time.
SOTA_TODO = None  # sentinel: fill from leaderboards, cite source + date


def mmlu_pro() -> dict:
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    answers = ds["answer"]           # option letter
    n_options = [len(o) for o in ds["options"]]
    result = accuracy_floor(list(answers))
    return {
        "benchmark": "MMLU-Pro",
        "split": "test",
        "metric": "mcq_accuracy",
        "n": len(answers),
        "floor": result.floor,
        "policy": result.policy,
        "uniform_random_floor": sum(1 / k for k in n_options) / len(n_options),
        "answer_distribution": dict(Counter(answers)),
        "sota": SOTA_TODO,
    }


def gpqa() -> dict:
    from datasets import load_dataset
    # gpqa_diamond: the subset essentially all published SOTA figures report,
    # so floor and SOTA come from the same split. Answers are randomized
    # per-instance by the harness, so the exploitable statistic is the option
    # position; the modal answer text is far below 1/4 (2.5% on diamond), so
    # the uniform guess over four options dominates.
    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    answers = [row["Correct Answer"].strip() for row in ds]
    result = exact_match_floor(answers)
    return {
        "benchmark": "GPQA (diamond)",
        "split": "train (the released eval set)",
        "metric": "mcq_accuracy",
        "n": len(answers),
        "floor": max(result.floor, 0.25),  # random guess over 4 options dominates
        "policy": "max(modal answer, uniform 1/4 guess); harness-shuffled options",
        "modal_answer_rate": result.floor,
        "sota": SOTA_TODO,
    }


def gsm8k() -> dict:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    finals = [row["answer"].rsplit("#### ", 1)[-1].strip().replace(",", "")
              for row in ds]
    em = exact_match_floor(finals)
    p8 = pass_at_k_floor(finals, k=8)
    counts = Counter(finals)
    top8 = counts.most_common(8)
    # The full histogram, not just the top three: the pass@8 floor, its support
    # size, and the statelessness pricing of App. A.6 are all functions of it,
    # and none of them was recheckable offline while only the mode was stored.
    return {
        "benchmark": "GSM8K",
        "split": "test",
        "metric": "exact_match",
        "n": len(finals),
        "floor": em.floor,
        "policy": em.policy,
        "top_answers": counts.most_common(3),
        "pass_at_8_stochastic_floor": p8.floor,
        "pass_at_8_support_size": p8.details.get("support_size"),
        "top_8_coverage": sum(c for _, c in top8) / len(finals),
        "top_8_answers": top8,
        "answer_distribution": dict(counts.most_common()),
        "sota": SOTA_TODO,
    }


def _gaia_key(answer: str) -> str:
    """Canonicalize one GAIA answer the way its official scorer compares them.

    The scorer is a quasi-exact match with three branches: a numeric answer is
    compared as a float, a comma-separated one elementwise, and anything else
    as a lowercased, punctuation-stripped string. Counting raw strings would
    understate the floor wherever two answers differ only by formatting.
    """
    import string

    def as_float(x: str):
        try:
            return float(x.replace(",", "").replace("$", "").replace("%", "").strip())
        except ValueError:
            return None

    def as_str(x: str) -> str:
        x = x.lower().strip().translate(str.maketrans("", "", string.punctuation))
        return " ".join(x.split())

    answer = answer.strip()
    if (f := as_float(answer)) is not None:
        return f"num:{f}"
    if "," in answer:
        parts = [e.strip() for e in answer.split(",")]
        return "list:" + "|".join(
            f"{g}" if (g := as_float(e)) is not None else as_str(e) for e in parts)
    return f"str:{as_str(answer)}"


def gaia() -> dict:
    """GAIA's floor, measured rather than assumed.

    Earlier drafts recorded this row as a structural zero on the argument that
    an open-ended answer space gives a constant answer measure zero. That is
    the misclassification checklist item 1 exists to catch: GAIA's answers are
    overwhelmingly small integers, so the modal answer is not rare. Gated on
    HF -- accept the terms and `huggingface-cli login` first.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("gaia-benchmark/GAIA",
                           "2023/validation/metadata.parquet", repo_type="dataset")
    answers = [str(a) for a in pd.read_parquet(path)["Final answer"]]
    result = exact_match_floor([_gaia_key(a) for a in answers])
    modal = Counter(a.strip() for a in answers).most_common(1)[0]
    return {
        "benchmark": "GAIA",
        "split": "validation",
        "metric": "exact_match",
        "n": len(answers),
        "floor": result.floor,
        "policy": f"constant modal answer {modal[0]!r} ({modal[1]}/{len(answers)})",
        "provenance": "analytic, from the gated validation answer file",
        "sota": SOTA_TODO,
    }


# SWE-bench Lite's test split: 300 task instances.
SWE_BENCH_LITE_N = 300


def positive_controls() -> list[dict]:
    """SWE-bench Lite: a structural lower bound, not a structural zero.

    The empty patch resolves nothing, but Definition 2 lets the constant be
    chosen with the labels in hand, and a constant policy that emits one
    task's gold patch resolves that task. So the floor is at least 1/n. Nothing
    here bounds it from above, which is why the row carries `floor_bound`.
    """
    return [
        {
            "benchmark": "SWE-bench Lite",
            "split": "test",
            "metric": "resolved_rate",
            "n": SWE_BENCH_LITE_N,
            "floor": 1 / SWE_BENCH_LITE_N,
            "floor_bound": "lower",
            "empty_patch_score": 0.0,
            "policy": "constant gold patch of one task (resolves that task); "
                      "the empty patch resolves nothing",
            "provenance": "structural lower bound 1/n (positive control); "
                          "not bounded above",
            "sota": SOTA_TODO,
        },
    ]


# WebArena's floor is a count over released task configs, not a label-file
# statistic, so it gets its own reader rather than sitting in the constants
# list. The count was previously inlined as 36/812 with a comment asserting it
# had been counted; nothing performed the count, which made it the one entry in
# the analytic group with no executable derivation.
WEBARENA_RECORDED = {"n_unachievable": 36, "n": 812}


def count_webarena_unachievable(config_path: Path) -> dict:
    """Count unachievable tasks in config_files/test.raw.json.

    WebArena marks a task unachievable by giving it the reference answer
    "N/A", and its string evaluator special-cases that reference: instead of
    the usual LLM fuzzy match it requires the prediction to be "N/A" too. So a
    constant-"N/A" policy scores exactly the unachievable fraction.

    The marker lives under `eval.reference_answers`, but on the `fuzzy_match`
    key rather than `exact_match` -- match on the value across whichever key
    carries it, so the count does not silently go to zero if the harness moves
    the annotation between match types.
    """
    configs = json.loads(config_path.read_text())
    unachievable = [
        c for c in configs
        if any(str(v).strip() == "N/A"
               for v in ((c.get("eval") or {}).get("reference_answers") or {}).values())
    ]
    return {"n": len(configs), "n_unachievable": len(unachievable),
            "task_ids": [c["task_id"] for c in unachievable]}


def webarena(config_path: Path | None) -> dict:
    """Floor from the unachievable-task count, checkable without the upstream file.

    results/webarena_unachievable.json records the derived evidence -- the 36
    task_ids -- rather than vendoring the 880 KB upstream config, which is
    third-party data that would go stale silently. The ids are enough for a
    reader to audit the count against upstream, and enough for this script to
    produce the floor with no network access. Passing --webarena-config
    recounts from the real file and cross-checks the two.
    """
    evidence_path = RESULTS / "webarena_unachievable.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.exists() else None

    if config_path is not None and config_path.exists():
        counted = count_webarena_unachievable(config_path)
        n, n_na = counted["n"], counted["n_unachievable"]
        provenance = (f"recounted from {config_path.name} "
                      f"({n_na}/{n} tasks with an 'N/A' reference answer)")
        recounted = "recounted-from-upstream"
        if evidence is not None:
            agree = sorted(counted["task_ids"]) == sorted(evidence["task_ids"])
            provenance += ("; task ids agree with "
                           f"{evidence_path.name}" if agree else
                           f"; WARNING: task ids DIFFER from {evidence_path.name}")
            if not agree:
                print(f"[warn] recount disagrees with {evidence_path.name} -- "
                      "upstream configs may have changed")
    elif evidence is not None:
        n, n_na = evidence["n_tasks"], evidence["n_unachievable"]
        provenance = (f"count from recorded task ids in {evidence_path.name} "
                      f"({evidence['source']}, retrieved {evidence['retrieved']}); "
                      "pass --webarena-config to recount from upstream")
        recounted = "from-recorded-task-ids"
    else:
        n, n_na = WEBARENA_RECORDED["n"], WEBARENA_RECORDED["n_unachievable"]
        provenance = ("recorded constant; neither the upstream config nor "
                      "results/webarena_unachievable.json was available")
        recounted = "recorded-constant"
    return {
        "benchmark": "WebArena",
        "split": "test",
        "metric": "success_rate",
        "n": n,
        "floor": n_na / n,
        "n_unachievable": n_na,
        "recounted": recounted,
        "policy": f"constant 'N/A' answer; scores the unachievable-task fraction ({n_na}/{n})",
        "provenance": provenance + "; mechanism per Zhou et al. 2024",
        "sota": SOTA_TODO,
    }


# The entries this file is expected to contain, in the order they are written
# so that a re-run produces a minimal diff.
EXPECTED = ["MMLU-Pro", "GPQA (diamond)", "GSM8K",
            "SWE-bench Lite", "GAIA", "WebArena"]

# Fields this script does not own: collect_results.py merges them in from
# sota.json afterwards. A run that recomputes a floor must not drop them.
CARRIED = ("sota", "sota_model", "sota_source", "sota_as_of", "sota_verified")


def merge(fresh: dict, previous: dict) -> dict:
    """Fresh values win, except for fields this script does not own.

    Without this, recomputing a floor silently reverted its SOTA annotation to
    the null sentinel, because those fields are added downstream.
    """
    out = dict(fresh)
    for key in CARRIED:
        if out.get(key) is None and previous.get(key) is not None:
            out[key] = previous[key]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-download", action="store_true",
                        help="reuse the recorded entries for the downloaded "
                             "benchmarks instead of refetching them")
    parser.add_argument("--webarena-config", type=Path, default=None,
                        help="path to WebArena's config_files/test.raw.json; "
                             "without it the recorded count is reused and "
                             "flagged recounted=false")
    parser.add_argument("--allow-partial", action="store_true",
                        help="write even when some expected entries are missing")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "analytic_floors.json"

    # Start from what is already recorded and update in place. This file is a
    # merge target, not a fresh artifact: any single invocation can only
    # recompute part of it (downloads may be skipped, GPQA is gated, the
    # WebArena configs live upstream), and a whole-file overwrite silently
    # deleted whatever that invocation could not produce.
    previous = {e["benchmark"]: e for e in json.loads(out.read_text())} if out.exists() else {}
    current = dict(previous)

    if not args.skip_download:
        for fn in (mmlu_pro, gpqa, gsm8k, gaia):
            try:
                fresh = fn()
                current[fresh["benchmark"]] = merge(fresh, previous.get(fresh["benchmark"], {}))
                print(f"[ok] {fresh['benchmark']}: floor={fresh['floor']:.4f}")
            except Exception as e:  # gated datasets, network
                # Keep the recorded entry rather than replacing it with an
                # error stub; report the failure instead of encoding it.
                print(f"[fail] {fn.__name__}: {e}")
    else:
        print("[skip] downloads: reusing recorded MMLU-Pro / GPQA / GSM8K / GAIA entries")

    for entry in positive_controls():
        current[entry["benchmark"]] = merge(entry, previous.get(entry["benchmark"], {}))

    wa = webarena(args.webarena_config)
    current["WebArena"] = merge(wa, previous.get("WebArena", {}))
    print(f"[{wa['recounted']}] WebArena: "
          f"floor={wa['floor']:.4f} ({wa['n_unachievable']}/{wa['n']})")

    missing = [b for b in EXPECTED if b not in current]
    if missing:
        msg = f"missing expected entries: {', '.join(missing)}"
        if not args.allow_partial:
            raise SystemExit(
                f"refusing to write an incomplete {out.name} -- {msg}.\n"
                "Re-run with the downloads available, or pass --allow-partial "
                "if a short file is genuinely what you want."
            )
        print(f"[warn] {msg} (writing anyway: --allow-partial)")

    entries = ([current[b] for b in EXPECTED if b in current]
               + [e for b, e in current.items() if b not in EXPECTED])
    out.write_text(json.dumps(entries, indent=2, default=str))
    print(f"wrote {out} ({len(entries)} entries)")
    if any(e.get("sota") is None for e in entries):
        print("[note] some entries have no SOTA yet; run collect_results.py "
              "to merge sota.json before make_figure1.py")


if __name__ == "__main__":
    main()
