"""Recompute every derived number the paper quotes, and check it.

The floors themselves are covered by analytic_floors.py and the released
per-episode JSONL. This script covers the layer above them: the intervals,
bounds, ratios and simulations quoted in the results section and appendices,
which were previously hand-computed and therefore unauditable. Each check
prints the paper's printed value beside the recomputed one and PASS/FAIL, and
the script exits non-zero if any check fails.

Everything runs offline from released artifacts -- results/*.json and the
per-task results/*.jsonl -- with no downloads and no benchmark harnesses. The
tau-bench counting identity of appendix C.3 is checked here rather than in
tau_static_floor.py for exactly that reason: that script needs tau-bench
installed, this one does not.

Two conventions worth stating, because both were got wrong before this script
existed:

  * A BOUND is rounded UP when quoted. Rounding 0.16089 to 0.16 states a
    tighter inequality than the one proved.
  * A RATIO is not rounded to a nicer integer. 4.77x is not "5x".

Run: .venv/bin/python scripts/check_appendix_numbers.py
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

DELTA = 0.05  # the confidence level used throughout the appendices


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def wilson(successes: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """Wilson score interval -- what Appendix C.2 reports for ALFWorld."""
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def cohens_kappa(y: list[int], pred: list[int]) -> float:
    n = len(y)
    p_o = sum(a == b for a, b in zip(y, pred)) / n
    py, pp = sum(y) / n, sum(pred) / n
    p_e = py * pp + (1 - py) * (1 - pp)
    return 0.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def average_precision(labels: list[int]) -> float:
    tp, total = 0, 0.0
    for rank, y in enumerate(labels, 1):
        if y:
            tp += 1
            total += tp / rank
    return total / sum(labels)


def water_filling_pass_at_k(counts: dict, k: int) -> tuple[float, int]:
    """Optimal stochastic pass@k floor, solved from the KKT condition of A.6.

    Reimplemented here rather than imported from floorcheck: this script exists
    to check the paper's printed numbers independently, and a check that calls
    the same code that produced the number is not a check. Bisection on the
    multiplier lambda, with q_a = 1 - (lambda / (k m_a))^(1/(k-1)) clipped at 0,
    and lambda chosen so the q_a sum to 1.
    """
    m = sorted(counts.values(), reverse=True)
    n = sum(m)

    def qs(lam: float) -> list[float]:
        out = []
        for m_a in m:
            marginal_at_zero = m_a * k
            if marginal_at_zero <= lam:
                break  # sorted, so every later answer is out of the support too
            out.append(1.0 - (lam / marginal_at_zero) ** (1.0 / (k - 1)))
        return out

    lo, hi = 1e-12, float(max(m) * k)
    for _ in range(200):
        mid = (lo + hi) / 2
        total = sum(qs(mid))
        if total > 1.0:
            lo = mid
        else:
            hi = mid
    q = qs((lo + hi) / 2)
    floor = sum(m_a * (1 - (1 - q_a) ** k) for m_a, q_a in zip(m, q)) / n
    return floor, len(q)


def load(name: str):
    return json.loads((RESULTS / name).read_text())


def load_jsonl(name: str) -> list[dict]:
    return [json.loads(line)
            for line in (RESULTS / name).read_text().splitlines() if line.strip()]


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

CHECKS: list[tuple[str, str, object, object, str]] = []


def check(where: str, what: str, printed, computed, ok: bool, note: str = "") -> None:
    CHECKS.append((where, what, printed, computed, "PASS" if ok else "FAIL"))
    if note:
        CHECKS[-1] = CHECKS[-1][:4] + (("PASS" if ok else "FAIL") + f"  ({note})",)


def close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=None,
                        help="override Monte Carlo replicates (default: the "
                             "counts the paper quotes)")
    args = parser.parse_args()

    bfcl = load("bfcl_slice_counts.json")
    d = bfcl["derived"]
    n_pooled, n_pos = d["n_pooled"], d["call_correct"]
    n_neg = d["abstain_correct_pooled"]
    pi = n_pos / n_pooled

    # --- section 5: the pooled detection task -----------------------------
    check("sec 5", "pooled slice sizes 240+884 abstain / 16 call",
          "240 + 884 / 16",
          f"{bfcl['v4']['BFCL_v4_irrelevance']['records']} + "
          f"{bfcl['v4']['BFCL_v4_live_irrelevance']['records']} / "
          f"{bfcl['v4']['BFCL_v4_live_relevance']['records']}",
          (bfcl["v4"]["BFCL_v4_irrelevance"]["records"] == 240
           and bfcl["v4"]["BFCL_v4_live_irrelevance"]["records"] == 884
           and n_pos == 16))
    check("sec 5", "constant-abstain raw accuracy", 0.986,
          round(d["raw_accuracy_constant_abstain"], 5),
          close(d["raw_accuracy_constant_abstain"], 0.986, 5e-4))
    check("sec 5", "pooled imbalance ratio", "70:1", f"{d['imbalance_ratio']:.2f}:1",
          close(d["imbalance_ratio"], 70.25, 0.01))

    # --- section 5 / figure 1: GSM8K floor growth -------------------------
    analytic = {e["benchmark"]: e for e in load("analytic_floors.json")}
    g = analytic["GSM8K"]
    ratio = g["pass_at_8_stochastic_floor"] / g["floor"]
    check("sec 5 + Fig 1", "GSM8K pass@8 / pass@1 growth", "4.8x", f"{ratio:.2f}x",
          close(ratio, 4.8, 0.05),
          "a ratio, not rounded to a nicer integer")

    # The released answer histogram makes the pass@k appendix checkable offline:
    # recompute the floor and its support from the histogram rather than trust
    # the stored scalars, which the same run produced.
    hist = g["answer_distribution"]
    floor8, support8 = water_filling_pass_at_k(hist, k=8)
    check("app A.6", "GSM8K pass@8 floor from the released histogram", 0.1447,
          round(floor8, 4), close(floor8, 0.1447, 5e-5),
          "solved here from the KKT condition, not read from floorcheck")
    check("app A.6", "GSM8K pass@8 support size", 18, support8, support8 == 18)
    top8 = sum(sorted(hist.values(), reverse=True)[:8])
    check("app A.6", "GSM8K top-8 coverage (statelessness pricing)", "267/1319",
          f"{top8}/{g['n']}", top8 == 267 and g["n"] == 1319,
          f"the stateful variant would score {top8 / g['n']:.3f}")

    # --- section 5 + table 3: the panel partition -------------------------
    # The bimodality sentence counts pairs by hand, which is how an earlier
    # draft carried a stale count past a change of sort key. Recount from the
    # released floor files so the partition and the headroom tally are checked
    # rather than asserted. The GSM8K pass@8 row is a second pair on the same
    # label file, so it is added here rather than stored as its own entry.
    executed = load("executed_floors.json")
    pairs = [(e["benchmark"], e["floor"]) for e in load("analytic_floors.json")]
    pairs += [(e["benchmark"], e["floor"]) for e in executed]
    pairs.append(("GSM8K (pass@8)", g["pass_at_8_stochastic_floor"]))
    static_qa = {"MMLU-Pro", "GPQA (diamond)", "GSM8K", "GSM8K (pass@8)"}
    n_static = sum(1 for b, _ in pairs if b in static_qa)
    n_artifact = sum(1 for b, _ in pairs if "(stop)" in b)
    agentic = [(b, f) for b, f in pairs if b not in static_qa and "(stop)" not in b]
    check("sec 4 + Tab 3", "audited benchmark-metric pairs", 18, len(pairs),
          len(pairs) == 18)
    check("Tab 3", "partition: static QA / harness artifacts / agentic",
          "4 / 2 / 12", f"{n_static} / {n_artifact} / {len(agentic)}",
          (n_static, n_artifact, len(agentic)) == (4, 2, 12),
          "the three parts must sum to 18")
    # Section 5 groups by mechanism, not by a threshold. A threshold is the wrong
    # instrument here: the headroom values do not separate, so any cut lands inside
    # a dense region and the grouping it produces is an artifact of where it fell.
    # Check the two claims the text makes instead.
    low = [b for b, f in agentic if f >= 0.07]
    check("sec 5", "agentic pairs conceding >= 0.07", 3, len(low), len(low) == 3,
          ", ".join(sorted(low)))
    check("sec 5", "the rest concede under 0.07", 9,
          len(agentic) - len(low), len(agentic) - len(low) == 9)
    # --- section 5 + app C.3/C.5: constants built from the answer key -----
    # Definition 2's floor is a supremum over all degenerate policies, and these
    # constants need the labels. Recounted by scripts/label_aware_webarena.py and
    # scripts/executed/tau_label_aware.py; checked here offline.
    wl = load("webarena_label_aware.json")
    check("sec 5 + app C.5", "WebArena label-aware constant (172/812)", 0.212,
          round(wl["constant_answer_passes"] / wl["n_tasks"], 4),
          wl["n_tasks"] == 812 and wl["constant_answer_passes"] == 172
          and len(wl["passed_task_ids"]) == 172
          and f"{172 / 812:.3f}" == "0.212",
          "one answer joining the must_include references of the substring-only tasks")
    check("app C.5", "distinct must_include references joined", 308,
          wl["must_include_references_distinct"], wl["must_include_references_distinct"] == 308)
    check("app C.5", "same scorer reproduces the paper's constant N/A", "36/812",
          f"{wl['constant_NA_passes (paper\'s policy)']}/{wl['n_tasks']}",
          wl["constant_NA_passes (paper's policy)"] == 36)
    tl = load("tau_label_aware.json")
    static_meta = load("tau_static_floor.json")["envs"]
    for env, want, txt in (("airline", 20, "0.400"), ("retail", 11, "0.096")):
        e = tl[env]
        static = [r["never_write_passes"] for r in load_jsonl(f"tau_{env}_static_floor.jsonl")]
        superset = all(i in e["passes_task_indices"] for i, p in enumerate(static) if p)
        check("sec 5 + app C.3", f"tau-bench {env} label-aware constant", f"{want}/{e['n']}",
              f"{len(e['passes_task_indices'])}/{e['n']}",
              len(e["passes_task_indices"]) == want
              and e["label_aware_constant (package reward)"] == want
              and f"{want / e['n']:.3f}" == txt
              and e["task_set_sha256"] == static_meta[env]["task_set_sha256"] and superset,
              "package reward agrees; contains every never-write pass; same task set")
    la = {"WebArena": wl["constant_answer_passes"] / wl["n_tasks"],
          "tau-bench airline (transfer)": len(tl["airline"]["passes_task_indices"]) / tl["airline"]["n"],
          "tau-bench retail (transfer)": len(tl["retail"]["passes_task_indices"]) / tl["retail"]["n"]}
    under = [b for b, f in agentic if max(f, la.get(b, 0)) < 0.07]
    check("sec 5", "agentic pairs under 0.07 with label-aware constants", 7, len(under),
          len(under) == 7, "WebArena and tau-bench retail move above 0.07")

    # The text concedes that retail does not separate from the high group by
    # value; that concession is only honest if it stays true.
    by_val = sorted(1 - f for _, f in agentic)
    gaps = [(round(b - a, 4), round(a, 3)) for a, b in zip(by_val, by_val[1:])]
    biggest = max(gaps)
    check("sec 5", "largest headroom gap sits below tau-retail, not above it",
          "below 0.939", f"at {biggest[1]}", biggest[1] < 0.939,
          "so no value-based cut puts retail with the low group")

    # --- appendix B: kappa's McDiarmid bound ------------------------------
    # |kappa| <= (1/(1-p_e)) sqrt(2 ln(2/delta) / n), at a fair coin p_e = 1/2.
    kbound = (1 / (1 - 0.5)) * math.sqrt(2 * math.log(2 / DELTA) / n_pooled)
    check("app B", "kappa bound at realized p_e = 1/2 (rounded UP)", 0.161,
          round(kbound, 5), 0.161 >= kbound and close(kbound, 0.161, 1e-3),
          "a bound: quoting 0.16 would claim more than is proved")

    # p_e is itself random, so 0.161 is conditional on its realized value. The
    # unconditional statement needs a delta/2 split: McDiarmid on the numerator
    # at delta/2, Hoeffding on the policy's own rate at delta/2, then evaluate
    # the bound at the resulting upper confidence limit for p_e.
    t_num = math.sqrt(2 * math.log(2 / (DELTA / 2)) / n_pooled)
    t_q = math.sqrt(math.log(2 / (DELTA / 2)) / (2 * n_pooled))
    pe_hi = 0.5 + t_q  # fair coin against a fixed label marginal of 1/2
    kbound_uniform = t_num / (1 - pe_hi)
    check("app B", "kappa bound, delta/2 union over p_e (rounded UP)", 0.193,
          round(kbound_uniform, 5),
          0.193 >= kbound_uniform and close(kbound_uniform, 0.193, 2e-3),
          "the honest unconditional number beside the conditional 0.161")

    # --- appendix B / 3.3: balanced-accuracy Hoeffding band ---------------
    ba_band = 0.5 * sum(math.sqrt(math.log(4 / DELTA) / (2 * nc))
                        for nc in (n_pos, n_neg))
    check("app B + sec 3.3", "balanced-accuracy band", "+-0.21",
          f"+-{ba_band:.4f}", close(ba_band, 0.21, 5e-3))

    # --- appendix C.2: ALFWorld -------------------------------------------
    # Only the *random* policy is sampled, so only it gets an interval. The
    # fixed policy is deterministic and the estimand is this 134-game split, so
    # the appendix reports 0/134 exactly; check that it is exactly 0.
    executed = {e["benchmark"]: e for e in load("executed_floors.json")}
    n_fixed = executed["ALFWorld (fixed)"]["n"]
    check("app C.2", f"ALFWorld fixed is exactly 0/{n_fixed} (no interval)",
          "0.000", f"{executed['ALFWorld (fixed)']['floor']:.4f}",
          executed["ALFWorld (fixed)"]["floor"] == 0.0 and n_fixed == 134,
          "deterministic policy on a fixed split has no sampling variance")
    rand = executed["ALFWorld (random)"]
    n_rand, k_rand, reps = rand["n"], rand["successes"], rand["replicates"]
    check("app C.2", "ALFWorld random replicates (checklist item 4 wants >= 20)",
          ">= 20", reps, reps >= 20 and n_rand == reps * rand["episodes_per_seed"])
    lo, hi = wilson(k_rand, n_rand)
    check("app C.2", f"ALFWorld random Wilson 95% ({k_rand}/{n_rand})",
          "[0.0015, 0.0059]", f"[{lo:.4f}, {hi:.4f}]",
          close(lo, 0.0015, 5e-5) and close(hi, 0.0059, 5e-5))
    single = rand["successes_per_seed"]["0"] / rand["episodes_per_seed"]
    check("app C.2", "single-seed floor overstated the 20-replicate one", "2.5x",
          f"{single / rand['floor']:.1f}x", close(single / rand["floor"], 2.5, 0.05),
          "seed 0 caught the success; 14 of 20 seeds score zero")

    # --- appendix C.3: the tau-bench counting identity ---------------------
    # The appendix claims the never-write predicate reproduces the executed
    # floor exactly, per task. Both sides are released as JSONL, so the claim
    # is checkable here without tau-bench installed -- which is the whole point
    # of persisting the predicate. Recomputed from the two JSONL files rather
    # than read from tau_static_floor.json, whose counts were written by the
    # same run that produced the predicate and so cannot corroborate it.
    for envname, floor_key, printed_floor, n_expected in (
            ("airline", "tau-bench airline (transfer)", 0.380, 50),
            ("retail", "tau-bench retail (transfer)", 0.061, 115)):
        static = [r["never_write_passes"]
                  for r in load_jsonl(f"tau_{envname}_static_floor.jsonl")]
        measured = [r["reward"] > 0
                    for r in load_jsonl(f"tau_{envname}_transfer.jsonl")]
        agree = sum(s == m for s, m in zip(static, measured))
        check("app C.3", f"tau-bench {envname} static floor ({sum(static)}/{len(static)})",
              f"{printed_floor:.3f}", f"{sum(static) / len(static):.4f}",
              len(static) == n_expected
              and close(sum(static) / len(static), printed_floor, 5e-4),
              "counting identity, no harness")
        check("app C.3", f"tau-bench {envname} per-task agreement",
              f"{n_expected}/{n_expected}", f"{agree}/{len(static)}",
              agree == len(static) == len(measured) == n_expected)
        # The identity is only interesting if it lands on the executed floor
        # the results table quotes, so tie the two together explicitly.
        recorded = executed[floor_key]["floor"]
        check("app C.3", f"tau-bench {envname} static == executed floor",
              f"{recorded:.4f}", f"{sum(static) / len(static):.4f}",
              close(sum(static) / len(static), recorded, 1e-9))

    # --- appendix A.4: E[AP] under random ranking exceeds pi --------------
    reps_ap = args.replicates or 40_000
    for (n, npos, printed_rel) in ((n_pooled, n_pos, 40), (1000, 100, 6)):
        rng = random.Random(0)
        labels = [1] * npos + [0] * (n - npos)
        total = 0.0
        for _ in range(reps_ap):
            rng.shuffle(labels)
            total += average_precision(labels)
        exp = total / reps_ap
        rel = 100 * (exp - npos / n) / (npos / n)
        check("app A.4", f"E[AP] excess at n={n}, pi={npos/n:.4f}",
              f"+{printed_rel}%", f"+{rel:.1f}%", close(rel, printed_rel, 1.0))
        if n == n_pooled:
            check("app A.4", f"E[AP] at n={n}", 0.0197, round(exp, 4),
                  close(exp, 0.0197, 5e-4))

    # --- appendix B: kappa's finite-sample skew ---------------------------
    reps_k = args.replicates or 30_000
    rng = random.Random(0)
    y = [1] * n_pos + [0] * (n_pooled - n_pos)
    vals = [cohens_kappa(y, [1 if rng.random() < pi else 0 for _ in range(n_pooled)])
            for _ in range(reps_k)]
    check("app B", "kappa median under marginal matching q=pi", -0.013,
          round(statistics.median(vals), 4),
          close(statistics.median(vals), -0.013, 1e-3))
    check("app B", "kappa mean under marginal matching q=pi", 1.3e-4,
          f"{statistics.mean(vals):+.1e}",
          abs(statistics.mean(vals)) < 1e-3, "chance, as immunity requires")
    rng = random.Random(0)
    fair = [cohens_kappa(y, [1 if rng.random() < 0.5 else 0 for _ in range(n_pooled)])
            for _ in range(reps_k)]
    check("app B", "kappa median under the fair coin (contrast)", "0",
          round(statistics.median(fair), 4),
          close(statistics.median(fair), 0.0, 2e-3),
          "the left shift belongs to q=pi, not to kappa at this n")

    # --- appendix A.3: macro-F1's plug-in is an upper bound at finite n ----
    # Table 1 puts macro-F1 in Band B at 1/K. The maximizer q = p is randomized,
    # so the ratio-of-expectations 1/K is not automatically E[macro-F1]. The row
    # is only safe if the gap has the conservative sign, so measure it.
    reps_f1 = args.replicates or 20_000
    for (k_cls, n_items) in ((3, 60), (3, 300), (5, 200), (10, 1140), (3, 1140)):
        rng = random.Random(0)
        p = [1 / k_cls] * k_cls
        labels = [i % k_cls for i in range(n_items)]
        total = 0.0
        for _ in range(reps_f1 // 20):  # coarser: we need the sign, not 4 digits
            pred = [rng.randrange(k_cls) for _ in range(n_items)]
            f1s = []
            for c in range(k_cls):
                tp = sum(1 for a, b in zip(labels, pred) if a == c and b == c)
                fp = sum(1 for a, b in zip(labels, pred) if a != c and b == c)
                fn = sum(1 for a, b in zip(labels, pred) if a == c and b != c)
                f1s.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
            total += sum(f1s) / k_cls
        mean = total / (reps_f1 // 20)
        check("app A.3", f"E[macro-F1] <= 1/K at K={k_cls}, n={n_items}",
              f"<= {1 / k_cls:.4f}", f"{mean:.4f}", mean <= 1 / k_cls + 1e-3,
              "conservative in the direction a floor needs")
    del p  # marginal-matching q = p is the uniform draw used above

    # --- appendix C.4: GPQA's constant-position policy on a realized shuffle
    gpqa = load("gpqa_position_floor.json")
    reps_pos = args.replicates or 40_000
    rng = random.Random(0)
    maxima = []
    for _ in range(reps_pos):
        counts = [0] * gpqa["k"]
        for _ in range(gpqa["n"]):
            counts[rng.randrange(gpqa["k"])] += 1
        maxima.append(max(counts) / gpqa["n"])
    mean_pos = sum(maxima) / len(maxima)
    check("app C.4", "GPQA E[max_j p_hat_j] over shuffles (n=198, K=4)", 0.287,
          round(mean_pos, 4), close(mean_pos, 0.287, 2e-3),
          "a realization band around 0.250, not a higher floor")
    check("app C.4", "recorded gpqa_position_floor.json agrees", 0.287,
          round(gpqa["best_position"]["mean"], 4),
          close(gpqa["best_position"]["mean"], mean_pos, 3e-3))
    # Section 5 now quotes the band's upper end as well as its mean.
    p95 = sorted(maxima)[int(0.95 * len(maxima))]
    check("sec 5 + app C.4", "GPQA 95th percentile of max_j p_hat_j", 0.318,
          round(gpqa["best_position"]["p95"], 4),
          close(gpqa["best_position"]["p95"], 0.318, 5e-4) and close(p95, 0.318, 6e-3))

    # --- section 5 / table 2: headroom and the airline capture ratio -------
    sota = load("sota.json")
    airline = executed["tau-bench airline (transfer)"]
    cr_hal = airline["floor"] / sota["tau-bench-airline"]["sota"]
    cr_board = (airline["floor"]
                / sota["tau-bench-airline"]["frozen_board_cross_check"]["sota"])
    check("sec 5 + app C.6", "airline CR at the quoted (HAL) ceiling", 0.68,
          round(cr_hal, 4), close(cr_hal, 0.68, 5e-3))
    check("sec 5 + app C.6", "airline CR range over attributable ceilings",
          "[0.68, 0.83]", f"[{cr_hal:.2f}, {cr_board:.2f}]",
          close(cr_hal, 0.68, 5e-3) and close(cr_board, 0.83, 5e-3),
          "0.700 excluded: a tau2-bench figure, see sota.json")
    for bench, printed_headroom in (("tau-bench airline (transfer)", 0.620),
                                    ("BFCL (irrelevance)", 0.000),
                                    ("ALFWorld (random)", 0.997)):
        hr = 1.0 - executed[bench]["floor"]
        check("table 2", f"headroom, {bench}", f"{printed_headroom:.3f}",
              f"{hr:.4f}", close(hr, printed_headroom, 5e-4))

    # --- the reporting census (S2 + app D) --------------------------------
    # The census is the one claim in the paper that rests on reading papers
    # rather than on running code, so what can be machine-checked is the
    # arithmetic over the coding sheet: that every framed paper is coded, and
    # that the printed fractions are the ones the sheet actually supports.
    census = json.loads((RESULTS / "reporting_census.json").read_text())
    papers, res = census["papers"], census["results"]
    coded = [p for p in papers if p["a_reports"] != "unresolved"]
    check("app D", "census frame is fully coded (no unresolved entries)",
          "29/29", f"{len(coded)}/{len(papers)}",
          len(coded) == len(papers) == 29,
          "tau-bench, OSWorld and AgentBoard were hand-coded from the arXiv PDFs")

    reports = [p["benchmark"] for p in coded if p["a_reports"] == "yes"]
    check("sec 1 + app D", "papers reporting a numeric oblivious baseline",
          "1/29", f"{len(reports)}/{len(coded)} ({', '.join(reports)})",
          reports == ["GPQA"])
    check("app D", "computed on the released split / normalized against",
          "0/29, 0/29",
          f"{sum(p['c_computed'].startswith('computed') for p in coded)}/{len(coded)},"
          f" {sum(p['d_normalized'] == 'yes' for p in coded)}/{len(coded)}",
          not any(p["c_computed"].startswith("computed") or p["d_normalized"] == "yes"
                  for p in coded))

    rationale = sorted(p["benchmark"] for p in coded
                       if p["b_location"] == "design rationale only")
    check("sec 2 + app D", "oblivious channel as design rationale, no score",
          "4/29", f"{len(rationale)}/{len(coded)}",
          rationale == ["MINT", "MMLU-Pro", "OSWorld", "WebArena"],
          ", ".join(rationale))
    for key, want in (("a_reports_numeric_oblivious_baseline", 1),
                      ("e_chance_rate_used_as_design_rationale_without_a_score", 4)):
        row = res[key]
        check("app D", f"results block agrees with the sheet ({key.split('_')[0]})",
              f"{row['count']}/{row['of']}", f"{want}/{len(coded)}",
              row["count"] == want and row["of"] == len(coded))

    # OSWorld's floor is derived from that paper's stated reward rule and task
    # counts, not audited here, so it gets no Table 2 row -- but app D quotes
    # the number, which makes it ours to keep true.
    osw = next(p for p in papers if p["benchmark"] == "OSWorld")["derived_floor_not_reported"]
    check("app D", "OSWorld constant-FAIL floor (30 infeasible of 369)", 0.081,
          f"{30 / 369:.4f}", close(osw["floor"], 30 / 369, 5e-5),
          "structural: fixed by the reward rule, not by a label file")
    check("app D", "OSWorld capture ratio vs its best reported agent", 0.66,
          f"{(30 / 369) / osw['best_reported_agent']:.3f}",
          close((30 / 369) / osw["best_reported_agent"], 0.66, 5e-3),
          "12.24% best model; not rounded to a nicer number")

    # --- section 4 + appendix C.5: GAIA is measured, not structural -------
    gaia = analytic["GAIA"]
    check("sec 4 + Tab 3", "GAIA exact-match floor (was a structural 0)", 0.042,
          round(gaia["floor"], 4), close(gaia["floor"], 0.042, 5e-4),
          f"modal answer over n={gaia['n']}; the draft it replaces guessed >= 0.006")
    check("sec 4", "GAIA floor exceeds the 1/n bound an earlier draft quoted",
          "7x", f"{gaia['floor'] / (1 / gaia['n']):.1f}x",
          gaia["floor"] > 1 / gaia["n"] and close(gaia["floor"] / (1 / gaia["n"]), 7.0, 0.5))

    # --- section 5 + tables 2-3: SWE-bench Lite is a lower bound, not a zero
    # The empty patch scores 0, but a constant policy that emits one task's gold
    # patch resolves that task, so the floor is at least 1/n. Nothing bounds it
    # from above, so the table prints the bound with its direction.
    swe = analytic["SWE-bench Lite"]
    check("sec 5 + Tab 2", "SWE-bench Lite floor is the 1/n lower bound",
          ">=0.003", f"{swe['floor']:.4f} ({swe['floor_bound']})",
          swe["n"] == 300 and close(swe["floor"], 1 / 300, 1e-12)
          and swe["floor_bound"] == "lower" and swe["empty_patch_score"] == 0.0,
          "one task's gold patch resolves it; the empty patch's 0 is not the floor")
    check("Tab 2", "SWE-bench Lite headroom and capture ratio", "<=0.997, >=0.005",
          f"{1 - swe['floor']:.4f}, {swe['floor'] / swe['sota']:.4f}",
          f"{1 - swe['floor']:.3f}" == "0.997" and swe["floor"] / swe["sota"] >= 0.005)

    # --- appendix C.3/C.5: the two assumptions that are now discharged ----
    tau_meta = load("tau_static_floor.json")["envs"]
    raised = {e: v["replay_actions_raised"] for e, v in tau_meta.items()}
    check("app C.3", "ground-truth replay actions that raise", 0,
          ", ".join(f"{e}={n}" for e, n in sorted(raised.items())),
          all(n == 0 for n in raised.values()),
          "zero makes the never-write predicate assumption-free, not merely conservative")
    wa = load("webarena_unachievable.json")["eval_types_verified"]
    types = wa["per_task_eval_types"]
    check("app C.5", "N/A tasks whose only eval_type is string_match",
          "36/36", f"{sum(v == ['string_match'] for v in types.values())}/{len(types)}",
          wa["all_string_match_only"] and len(types) == 36 and not wa["with_program_html"],
          "nothing else has to pass, so the constant-N/A policy scores all 36")

    # --- section 2: both census denominators --------------------------
    ao = res["a_reports_agentic_only"]
    check("sec 2 + app D", "papers reporting, agentic frame members only",
          "0/26", f"{ao['count']}/{ao['of']}",
          ao["count"] == 0 and ao["of"] == 26,
          "the one yes (GPQA) is a static-QA release")

    # --- table 1 + appendix A.4: the tool's bound field matches the paper -
    # A half-fix here is worse than none: the caption tells a reader what
    # floorcheck emits, so the claim and the code have to move together.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "floorcheck"))
    from floorcheck import closed_forms as cf  # noqa: E402
    bounds = {"auprc": cf.auprc_floor(0.1).bound,
              "macro_f1": cf.macro_f1_floor(num_classes=3).bound,
              "cohens_kappa": cf.cohens_kappa_floor().bound,
              "mcc": cf.mcc_floor().bound,
              "roc_auc": cf.roc_auc_floor().bound,
              "balanced_accuracy": cf.balanced_accuracy_floor(2).bound}
    want = {"auprc": "lower", "macro_f1": "upper", "cohens_kappa": "expectation",
            "mcc": "expectation", "roc_auc": "exact",
            "balanced_accuracy": "exact"}
    check("Tab 1 + app A.4", "floorcheck bound field per metric",
          "lower / upper / expectation x2 / exact x2",
          " / ".join(f"{k}={v}" for k, v in sorted(bounds.items())),
          bounds == want,
          "macro-F1's 1/K is a plug-in the i.i.d. draw falls short of; "
          "kappa and MCC divide by a policy-dependent denominator")

    # --- appendix D: the verification pass --------------------------------
    ver = load("census_verification.json")
    check("app D", "second pass agrees with the first on (a)", "29/29",
          f"{ver['agreement_on_a']['agree']}/{ver['agreement_on_a']['of']}",
          ver["agreement_on_a"]["agree"] == len(coded)
          and ver["agreement_on_a"]["of"] == len(coded))
    check("app D", "kappa deliberately not reported", "not reported",
          "not reported", ver["kappa_reported"] is False,
          "one yes in 29 puts p_e within 1e-3 of 1; see app B")
    corrected = [c["benchmark"] for c in ver["corrections_produced"]]
    check("app D", "corrections the verification pass produced", "MINT",
          ", ".join(corrected) or "none", corrected == ["MINT"],
          "the correction that moves (e) from 3 to 4")

    # --- report -----------------------------------------------------------
    width = max(len(c[1]) for c in CHECKS)
    failures = 0
    print(f"{'where':<16}{'quantity':<{width + 2}}{'paper':>12}{'computed':>14}   status")
    print("-" * (16 + width + 2 + 12 + 14 + 12))
    for where, what, printed, computed, status in CHECKS:
        if status.startswith("FAIL"):
            failures += 1
        print(f"{where:<16}{what:<{width + 2}}{str(printed):>12}{str(computed):>14}   {status}")
    print()
    print(f"{len(CHECKS) - failures}/{len(CHECKS)} checks pass")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
