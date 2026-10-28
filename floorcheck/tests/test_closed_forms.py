"""Closed forms checked against brute force / simulation."""

import contextlib
import io
import itertools
import json
import math
import random

import pytest

from floorcheck import cli
from floorcheck import closed_forms as cf


def _run_cli_rows(argv: list[str]) -> list[dict]:
    """Run the CLI and parse its stream of JSON objects back out."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(argv)
    decoder, text, rows, idx = json.JSONDecoder(), buf.getvalue(), [], 0
    while idx < len(text):
        if text[idx].isspace():
            idx += 1
            continue
        obj, idx = decoder.raw_decode(text, idx)
        rows.append(obj)
    return rows


def _run_cli(argv: list[str]) -> dict:
    rows = _run_cli_rows(argv)
    assert len(rows) == 1
    return rows[0]


def test_accuracy_floor_is_best_constant():
    labels = ["A"] * 60 + ["B"] * 30 + ["C"] * 10
    result = cf.accuracy_floor(labels)
    assert result.floor == pytest.approx(0.6)
    assert result.details["mode"] == "A"
    # brute force over constants
    best = max(sum(1 for y in labels if y == c) / len(labels) for c in set(labels))
    assert result.floor == pytest.approx(best)


def test_accuracy_floor_accepts_counts():
    assert cf.accuracy_floor({"A": 60, "B": 40}).floor == pytest.approx(0.6)


def test_binary_f1_floor_matches_confusion_matrix():
    pi = 0.35
    n = 1000
    tp, fp, fn = pi * n, (1 - pi) * n, 0
    f1 = 2 * tp / (2 * tp + fp + fn)
    assert cf.binary_f1_floor(pi).floor == pytest.approx(f1)
    assert cf.binary_f1_floor(0.5).floor == pytest.approx(2 / 3)


def test_precision_recall_floors():
    assert cf.precision_floor(0.2).floor == pytest.approx(0.2)
    assert cf.recall_floor(0.2).floor == 1.0


def test_rate_validation():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            cf.binary_f1_floor(bad)


def test_pass_at_1_equals_mode():
    labels = ["x"] * 50 + ["y"] * 30 + ["z"] * 20
    result = cf.pass_at_k_floor(labels, k=1)
    assert result.floor == pytest.approx(0.5)


def test_pass_at_k_deterministic_flat_in_k():
    labels = ["x"] * 50 + ["y"] * 50
    for k in (1, 4, 16):
        assert cf.pass_at_k_floor(labels, k=k, stochastic=False).floor == pytest.approx(0.5)


def test_pass_at_k_stochastic_beats_deterministic_and_grows_with_k():
    labels = ["x"] * 50 + ["y"] * 30 + ["z"] * 20
    floors = [cf.pass_at_k_floor(labels, k=k).floor for k in (1, 2, 4, 8)]
    assert all(b >= a - 1e-12 for a, b in zip(floors, floors[1:]))
    assert floors[-1] > 0.5  # beats the mode-only policy


def test_waterfilling_matches_grid_search():
    # two answers, k=2: maximize p1*(1-(1-q)^2) + p2*(1-q^2) over q in [0,1]
    freqs = {"a": 0.6, "b": 0.4}
    k = 2
    q = cf.optimal_guess_distribution(freqs, k)
    assert sum(q.values()) == pytest.approx(1.0, abs=1e-6)

    def objective(qa):
        return 0.6 * (1 - (1 - qa) ** k) + 0.4 * (1 - qa ** k)

    best_grid = max(objective(i / 10000) for i in range(10001))
    achieved = objective(q["a"])
    assert achieved == pytest.approx(best_grid, abs=1e-6)


def test_waterfilling_drops_rare_answers():
    # with many answers and small k, rare answers get zero mass
    freqs = {"a": 0.7, "b": 0.2, "c": 0.05, "d": 0.05}
    q = cf.optimal_guess_distribution(freqs, k=2)
    assert q["a"] > q["b"] > 0
    assert sum(q.values()) == pytest.approx(1.0, abs=1e-6)


def test_macro_f1_floor_is_one_over_k():
    labels = ["A"] * 80 + ["B"] * 15 + ["C"] * 5  # heavy imbalance, K = 3
    result = cf.macro_f1_floor(labels)
    assert result.floor == pytest.approx(1 / 3)
    # the best constant policy is strictly worse than marginal matching
    assert result.details["constant_policy_floor"] < result.floor


def test_macro_f1_marginal_matching_beats_grid_on_simplex():
    # K=2: macroF1(q) = (1/2)[2*p1*q/(p1+q) + 2*p2*(1-q)/(p2+1-q)], max at q=p1
    p1, p2 = 0.8, 0.2

    def objective(q):
        return 0.5 * (2 * p1 * q / (p1 + q) + 2 * p2 * (1 - q) / (p2 + 1 - q))

    best_grid = max(objective(i / 10000) for i in range(10001))
    assert objective(p1) == pytest.approx(best_grid, abs=1e-6)
    assert objective(p1) == pytest.approx(0.5, abs=1e-9)


def test_macro_f1_needs_two_classes():
    with pytest.raises(ValueError):
        cf.macro_f1_floor(["A"] * 10)


def test_immune_floors():
    assert cf.balanced_accuracy_floor().floor == 0.5
    assert cf.cohens_kappa_floor().floor == 0.0
    assert cf.mcc_floor().floor == 0.0
    assert cf.roc_auc_floor().floor == 0.5
    assert all(r().immune for r in cf.IMMUNE_METRICS.values())


def test_roc_auc_immunity_by_simulation():
    # i.i.d. random scores independent of labels: Mann-Whitney AUC -> 0.5
    rng = random.Random(0)
    y = [rng.random() < 0.9 for _ in range(4000)]  # imbalanced labels
    s = [rng.random() for _ in y]  # input-oblivious scores
    pos = [si for si, yi in zip(s, y) if yi]
    neg = [si for si, yi in zip(s, y) if not yi]
    wins = sum(1 for sp in pos for sn in neg if sp > sn)
    ties = sum(1 for sp in pos for sn in neg if sp == sn)
    auc = (wins + 0.5 * ties) / (len(pos) * len(neg))
    assert auc == pytest.approx(0.5, abs=0.02)


def test_balanced_accuracy_immunity_by_simulation():
    # any constant policy: BA = 0.5 exactly; random policy: BA -> 0.5
    rng = random.Random(0)
    y = [rng.random() < 0.8 for _ in range(200_000)]
    yhat = [rng.random() < 0.3 for _ in range(200_000)]  # input-oblivious
    tp = sum(a and b for a, b in zip(y, yhat))
    fn = sum(a and not b for a, b in zip(y, yhat))
    tn = sum(not a and not b for a, b in zip(y, yhat))
    fp = sum(not a and b for a, b in zip(y, yhat))
    ba = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
    assert ba == pytest.approx(0.5, abs=0.01)


def test_capture_ratio_and_normalized_score():
    assert cf.capture_ratio(0.4, 0.8) == pytest.approx(0.5)
    assert cf.normalized_score(0.6, floor=0.2) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        cf.capture_ratio(0.4, 0.0)
    with pytest.raises(ValueError):
        cf.normalized_score(0.6, floor=1.2)


def test_headroom_and_terminal_diagnosis_at_floor_one():
    # Headroom needs no ceiling: it is defined on every pair.
    assert cf.headroom(0.38) == pytest.approx(0.62)
    assert cf.headroom(0.0) == pytest.approx(1.0)
    # A floor equal to the metric's maximum (BFCL's abstention slices) leaves
    # no scale to normalize against. That is a terminal diagnosis, not an
    # error: the metric should be replaced, so we report it rather than raise.
    assert cf.headroom(1.0) == pytest.approx(0.0)
    assert cf.normalized_score(0.95, floor=1.0) is None
    assert cf.recall_floor(0.3).floor == pytest.approx(1.0)
    assert cf.normalized_score(0.95, floor=cf.recall_floor(0.3).floor) is None


def test_pass_hat_k_is_flat_in_k_and_below_pass_at_k():
    # pass^k requires ALL k trials to pass, so extra samples never help a
    # degenerate policy; pass@k requires at least one, so they do.
    labels = ["a"] * 40 + ["b"] * 30 + ["c"] * 20 + ["d"] * 10
    floors = [cf.pass_hat_k_floor(labels, k=k).floor for k in (1, 2, 8, 64)]
    assert floors == [pytest.approx(0.4)] * 4
    assert cf.pass_hat_k_floor(labels, k=8).details["flat_in_k"] is True
    # the two metrics agree at k=1 and diverge above it
    assert cf.pass_at_k_floor(labels, k=1).floor == pytest.approx(floors[0])
    assert cf.pass_at_k_floor(labels, k=8).floor > cf.pass_hat_k_floor(labels, k=8).floor


def test_macro_f1_uses_declared_class_count():
    # Observed labels cover 2 of 4 declared classes. Inferring K from the split
    # would report 1/2; the declared set gives 1/4.
    labels = ["a"] * 50 + ["b"] * 50
    assert cf.macro_f1_floor(labels).floor == pytest.approx(0.5)
    r = cf.macro_f1_floor(labels, num_classes=4)
    assert r.floor == pytest.approx(0.25)
    assert r.details["observed_classes"] == 2
    assert r.details["unsupported_classes"] == 2
    with pytest.raises(ValueError):
        cf.macro_f1_floor(labels, num_classes=1)


def test_balanced_accuracy_is_immune_at_one_over_k():
    # Balanced accuracy is macro-averaged recall: an oblivious policy drawing
    # class c with probability q_c has recall_c = q_c, so the macro-average is
    # (1/K) * sum_c q_c = 1/K exactly -- immune for every K, not only K = 2.
    assert cf.balanced_accuracy_floor().floor == pytest.approx(0.5)
    assert cf.balanced_accuracy_floor(2).floor == pytest.approx(0.5)
    assert cf.balanced_accuracy_floor(4).floor == pytest.approx(0.25)
    assert cf.balanced_accuracy_floor(10).floor == pytest.approx(0.1)
    assert cf.balanced_accuracy_floor(10).immune is True
    with pytest.raises(ValueError):
        cf.balanced_accuracy_floor(1)
    # kappa and MCC do not depend on K at all.
    assert cf.cohens_kappa_floor().floor == 0.0
    assert cf.mcc_floor().floor == 0.0


def test_multiclass_balanced_accuracy_matches_simulation():
    # Empirical check that an oblivious multi-class policy lands on 1/K.
    rng = random.Random(0)
    K, n = 5, 40000
    labels = [rng.choices(range(K), weights=[0.5, 0.2, 0.15, 0.1, 0.05])[0]
              for _ in range(n)]
    preds = [rng.choices(range(K), weights=[0.7, 0.1, 0.1, 0.05, 0.05])[0]
             for _ in range(n)]
    recalls = []
    for c in range(K):
        idx = [i for i, y in enumerate(labels) if y == c]
        recalls.append(sum(preds[i] == c for i in idx) / len(idx))
    assert sum(recalls) / K == pytest.approx(1 / K, abs=0.01)


def test_band_and_label_free_match_the_paper_table():
    # Band A floors move with the label distribution; Band B floors are fixed
    # by the metric and its declared class set. label_free is about the
    # *policy*, not the floor value: macro-F1's floor is 1/K but attaining it
    # means matching the label marginals, so its policy reads labels.
    labels = ["a"] * 6 + ["b"] * 4
    band_a = [cf.accuracy_floor(labels), cf.mcq_accuracy_floor(labels),
              cf.exact_match_floor(labels), cf.binary_f1_floor(0.3),
              cf.precision_floor(0.3), cf.auprc_floor(0.3),
              cf.pass_at_k_floor(labels, k=4), cf.pass_hat_k_floor(labels, k=4)]
    band_b = [cf.recall_floor(0.3), cf.macro_f1_floor(labels),
              cf.balanced_accuracy_floor(4), cf.cohens_kappa_floor(),
              cf.mcc_floor(), cf.roc_auc_floor()]
    assert [r.band for r in band_a] == ["A"] * len(band_a)
    assert [r.band for r in band_b] == ["B"] * len(band_b)

    # Every immune metric is attained by any policy, so it is label-free.
    assert all(cf.IMMUNE_METRICS[m]().label_free for m in ("cohens_kappa", "mcc", "roc_auc"))
    assert cf.balanced_accuracy_floor(4).label_free is True
    # All-positive and random-ranking policies need no labels to run.
    assert cf.recall_floor(0.3).label_free is True
    assert cf.auprc_floor(0.3).label_free is True
    # Modal-answer and marginal-matching policies do.
    assert cf.accuracy_floor(labels).label_free is False
    assert cf.macro_f1_floor(labels).label_free is False
    assert cf.pass_at_k_floor(labels, k=4).label_free is False


def test_cli_sota_accepts_a_contested_range():
    # Checklist item 6 asks for a range when the ceiling is contested, so a
    # scalar alone cannot express tau-bench airline. Dividing by the larger
    # ceiling gives the smaller ratio, so the ratio range comes out reversed.
    assert cli._parse_sota(None) is None
    assert cli._parse_sota("0.7") == (0.7,)
    assert cli._parse_sota("0.46:0.70") == (0.46, 0.70)
    out = _run_cli(["compute", "--metric", "precision", "--pi", "0.38",
                    "--sota", "0.46:0.70", "--sota-as-of", "2024-10"])
    assert out["capture_ratio_range"] == pytest.approx([0.542857, 0.826087], abs=1e-5)
    assert out["sota_as_of"] == "2024-10"
    assert "capture_ratio" not in out
    for bad in ("0.4:0.5:0.6", "0.70:0.46", "not-a-number"):
        with pytest.raises(SystemExit):
            cli._parse_sota(bad)


def test_report_refuses_to_broadcast_one_sota_across_metrics(tmp_path):
    # Definition 2 requires a ceiling on the same split *and metric*. `report`
    # spans several metrics, so a single --sota must name its row rather than
    # producing a capture ratio for every one of them.
    path = tmp_path / "labels.json"
    path.write_text(json.dumps(["a"] * 6 + ["b"] * 3 + ["c"]))
    with pytest.raises(SystemExit):
        cli.main(["report", str(path), "--sota", "0.82"])
    with pytest.raises(SystemExit):
        cli.main(["report", str(path), "--score", "0.82"])
    with pytest.raises(SystemExit):
        cli.main(["report", str(path), "--sota", "0.82", "--for-metric", "nonesuch"])

    rows = _run_cli_rows(["report", str(path), "--sota", "0.82",
                          "--for-metric", "accuracy"])
    by_metric = {r["metric"]: r for r in rows}
    assert "capture_ratio" in by_metric["accuracy"]
    assert all("capture_ratio" not in r for m, r in by_metric.items() if m != "accuracy")
    # The report carries pass^k beside pass@k, where the contrast is visible:
    # pass@k grows with k while pass^k stays at p_mode.
    assert by_metric["pass^8"]["floor"] == pytest.approx(by_metric["accuracy"]["floor"])
    assert by_metric["pass@8"]["floor"] > by_metric["pass^8"]["floor"]


def test_macro_f1_needs_no_label_file():
    # Table 1 files macro-F1 under Band B: "state it once, no label file
    # needed". The tool has to honour that, or the checklist misinstructs.
    r = cf.macro_f1_floor(num_classes=10)
    assert r.floor == pytest.approx(0.1)
    assert r.band == "B"
    # The floor is label-free; the policy attaining it is not, since matching
    # the marginals needs them. Those are different questions.
    assert r.label_free is False
    with pytest.raises(ValueError):
        cf.macro_f1_floor()  # neither labels nor a declared class count


def test_auprc_is_flagged_as_a_lower_bound():
    # pi is asymptotic: E[AP] exceeds it at finite n under imbalance, so a tool
    # that emitted pi unqualified would under-report the floor.
    r = cf.auprc_floor(0.014035)
    assert r.floor == pytest.approx(0.014035)
    assert r.bound == "lower"
    assert r.details["bound_direction"]
    # the label-count rows are exact finite-panel suprema
    assert cf.accuracy_floor(["a", "a", "b"]).bound == "exact"
    assert cf.binary_f1_floor(0.35).bound == "exact"
    assert cf.balanced_accuracy_floor(4).bound == "exact"


def test_macro_f1_is_flagged_as_an_upper_bound():
    # AUPRC's closed form errs low; macro-F1's errs high. 1/K is the plug-in
    # value of the marginal-matching draw, and the draw's expected score is
    # below it at finite n, so a tool that emitted 1/K as "exact" would
    # overstate what an i.i.d. oblivious policy collects.
    labels = ["a"] * 4 + ["b"] * 2
    r = cf.macro_f1_floor(labels)
    assert r.floor == pytest.approx(0.5)
    assert r.bound == "upper"
    assert "Jensen" in r.details["bound_direction"]
    assert cf.macro_f1_floor(num_classes=10).bound == "upper"
    # Exact expectation under q = p_hat, enumerating all 2^6 prediction vectors.
    q = {"a": 4 / 6, "b": 2 / 6}
    expected = 0.0
    for preds in itertools.product("ab", repeat=len(labels)):
        prob = math.prod(q[p] for p in preds)
        f1 = [2 * sum(y == p == c for y, p in zip(labels, preds))
              / (labels.count(c) + preds.count(c)) for c in "ab"]
        expected += prob * sum(f1) / 2
    assert expected < 0.5


def test_kappa_and_mcc_are_flagged_as_expectations():
    """The two immune rows whose denominators the policy moves.

    Balanced accuracy and ROC-AUC divide by counts the labels fix, so a
    constant policy lands on chance on every realization and so does the
    expectation. Kappa and MCC do not: their denominators depend on what the
    policy emitted, so the 0 survives only in expectation, up to the O(1/n)
    ratio bias of Proposition 1(b). Reporting those two as "exact" would claim
    more than the paper proves.
    """
    for r in (cf.cohens_kappa_floor(), cf.mcc_floor()):
        assert r.floor == 0.0
        assert r.immune is True
        assert r.bound == "expectation", r.metric
        assert "O(1/n)" in r.details["bound_direction"]
    # the fixed-denominator half of the immune family stays exact
    assert cf.roc_auc_floor().bound == "exact"
    assert cf.balanced_accuracy_floor(3).bound == "exact"
