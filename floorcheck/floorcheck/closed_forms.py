"""Closed-form degenerate-baseline floors, by metric family.

Every function answers one question: what is the best expected score an
input-oblivious policy (constant output, or i.i.d. draws from a fixed
distribution) can achieve on this metric, given only label statistics?

Notation follows the paper: pi = positive-class rate, p_c = frequency of
label c, p_mode = frequency of the modal label.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence


@dataclass
class FloorResult:
    """A computed floor plus the two facts the floor report asks about it.

    `band` is the paper's Table 1 band of the metric family: "A" when the floor
    moves with the label distribution, "B" when the metric and its declared
    class set fix it. Band C never appears here -- it is by definition the
    no-closed-form case, which this module cannot compute.

    `label_free` says whether the *maximizing policy* can be run without label
    access, which is a different question from whether the floor value needs
    labels: macro-F1's floor is 1/K, but attaining it means matching the label
    marginals, so the policy is not label-free.

    `bound` says how `floor` relates to the supremum of Definition 2, and
    takes four values. Most entries are "exact". AUPRC is "lower": pi is the
    asymptotic value and the finite-panel expectation of average precision
    exceeds it under heavy imbalance. Macro-F1 is "upper": 1/K is the plug-in
    value of the marginal-matching draw, and at finite n the expected
    macro-F1 of any i.i.d. draw falls short of its plug-in value, reaching 1/K
    only as the rarest class grows. Cohen's kappa and MCC are "expectation":
    both are ratios whose denominators the policy moves, so a constant policy
    scores 0 on every realization but a randomized one scores 0 only in
    expectation, up to the O(1/n) bias of a ratio estimator. A floor report
    that quoted any of the four as if it were the others would misstate
    exactly what this tool exists to surface, so the direction travels with
    the number rather than living in the paper's appendix.
    """

    metric: str
    floor: float
    policy: str
    immune: bool = False
    band: str = "A"
    label_free: bool = False
    bound: str = "exact"
    details: dict = field(default_factory=dict)

    def capture_ratio(self, sota: float) -> float:
        if sota <= 0:
            raise ValueError("SOTA score must be positive")
        return self.floor / sota


def _freqs(labels: Iterable | Mapping[object, int]) -> dict:
    """Normalize labels (sequence) or counts (mapping) to frequencies."""
    if isinstance(labels, Mapping):
        counts = dict(labels)
    else:
        counts = Counter(labels)
    n = sum(counts.values())
    if n == 0:
        raise ValueError("empty label set")
    return {k: v / n for k, v in counts.items()}


# ---------------------------------------------------------------------------
# Accuracy family: best constant = modal label, floor = p_mode.
# Covers plain accuracy, MCQ accuracy, and exact-match scoring; for MCQ
# position bias, pass position labels instead of option labels.
# ---------------------------------------------------------------------------

def accuracy_floor(labels: Iterable | Mapping[object, int]) -> FloorResult:
    freqs = _freqs(labels)
    mode, p_mode = max(freqs.items(), key=lambda kv: kv[1])
    return FloorResult(
        metric="accuracy",
        floor=p_mode,
        policy=f"constant prediction of modal label {mode!r}",
        band="A", label_free=False,
        details={"mode": mode, "num_classes": len(freqs)},
    )


def mcq_accuracy_floor(labels: Iterable | Mapping[object, int]) -> FloorResult:
    result = accuracy_floor(labels)
    result.metric = "mcq_accuracy"
    return result


def exact_match_floor(labels: Iterable | Mapping[object, int]) -> FloorResult:
    result = accuracy_floor(labels)
    result.metric = "exact_match"
    result.policy = "constant modal answer"
    return result


# ---------------------------------------------------------------------------
# Binary P/R/F1: all-positive policy. pi = positive rate.
# ---------------------------------------------------------------------------

def binary_f1_floor(pi: float) -> FloorResult:
    _check_rate(pi)
    return FloorResult(
        metric="binary_f1",
        floor=2 * pi / (1 + pi),
        policy="predict positive on every item (precision=pi, recall=1)",
        band="A", label_free=True,
        details={"positive_rate": pi},
    )


def precision_floor(pi: float) -> FloorResult:
    _check_rate(pi)
    return FloorResult(metric="precision", floor=pi, policy="predict all positive",
                       band="A", label_free=True)


def recall_floor(pi: float) -> FloorResult:
    _check_rate(pi)
    return FloorResult(metric="recall", floor=1.0, policy="predict all positive",
                       band="B", label_free=True)


def auprc_floor(pi: float) -> FloorResult:
    """Random-ranking AUPRC floor: pi, asymptotically, and a LOWER bound at finite n.

    A uniformly random ranking has expected precision pi at every fixed *rank*
    cutoff, so the PR curve is flat at height pi in expectation and the
    limiting AUPRC is pi. Average precision, the usual estimator, instead
    averages precision at the positions where positives land, and conditioning
    on a positive occupying a position raises the precision there: E[AP] > pi,
    by 40% relative at n = 1140, pi = 0.014. So pi is a lower bound on the
    floor, and the excess is largest under the heavy imbalance that motivates
    detection metrics in the first place.
    """
    _check_rate(pi)
    return FloorResult(
        metric="auprc",
        floor=pi,
        policy="uniformly random ranking (expected precision = pi at every fixed rank cutoff)",
        band="A", label_free=True, bound="lower",
        details={
            "bound_direction": "E[AP] exceeds pi at finite n; the excess grows "
                               "as pi falls (about +40% relative at n = 1140, "
                               "pi = 0.014, and +6% at n = 1000, pi = 0.10)",
            "guidance": "quote pi as a lower bound on the floor, not as the "
                        "floor, on rare-positive slices",
        },
    )


def _check_rate(pi: float) -> None:
    if not 0 < pi < 1:
        raise ValueError(f"positive rate must be in (0, 1), got {pi}")


# ---------------------------------------------------------------------------
# Multi-class F1. Micro-F1 (single-label) equals accuracy. Macro-F1's optimal
# input-oblivious policy matches the label marginals (q = p), whose plug-in
# score is 1/K; its expected score approaches that from below as n grows. The
# best constant (modal class m) gets only (2 p_m/(1+p_m))/K.
# ---------------------------------------------------------------------------

_MACRO_F1_UPPER = (
    "1/K is the plug-in value of the marginal-matching draw. At finite n its "
    "expected macro-F1 is lower: given how many items the policy assigns to a "
    "class, those items are a uniformly random subset, so that class's "
    "expected F1 is concave in the count and Jensen's inequality bounds it by "
    "the plug-in. It reaches 1/K only as n * min_c p_c grows."
)


def macro_f1_floor(
    labels: Iterable | Mapping[object, int] | None = None,
    num_classes: int | None = None,
) -> FloorResult:
    """Macro-F1 floor: at most 1/K, approached by marginal matching.

    The value is an upper bound (`bound="upper"`). 1/K is the plug-in score of
    i.i.d. draws matching the label marginals; their expected score falls
    short of it at finite n and approaches it as the rarest class grows.

    K is the number of classes macro-F1 averages over, which is the *declared*
    label set -- not necessarily the set observed in this split. A declared
    class with no support contributes F1 = 0 to the average, so inferring K
    from the observed labels overstates the floor. Pass num_classes when the
    declared set is larger than what the label file happens to contain.

    Macro-F1 is a Band B row: the floor is fixed by the declared class count,
    so `macro_f1_floor(num_classes=K)` with no label file returns it. Labels
    are optional and buy two things the constant alone does not: the observed
    class count, and the score of the best *constant* policy, which is
    strictly below 1/K and shows that the optimum is randomized. Note that
    while the floor is label-free, the policy attaining it is not -- matching
    the marginals needs them -- which is what `label_free=False` records.
    """
    if labels is None:
        if num_classes is None:
            raise ValueError("pass labels or num_classes")
        if num_classes < 2:
            raise ValueError("macro-F1 needs at least two classes")
        return FloorResult(
            metric="macro_f1",
            floor=1.0 / num_classes,
            policy="i.i.d. draws matching the label marginals (q = p)",
            band="B", label_free=False, bound="upper",
            details={"num_classes": num_classes,
                     "bound_direction": _MACRO_F1_UPPER,
                     "note": "declared class count only; no label file was read, "
                             "so the observed class count and the best constant "
                             "policy's score are not reported"},
        )
    freqs = _freqs(labels)
    observed = len(freqs)
    if observed < 2:
        raise ValueError("macro-F1 needs at least two observed classes")
    k = observed if num_classes is None else num_classes
    if k < observed:
        raise ValueError(
            f"num_classes={num_classes} is below the {observed} classes observed "
            "in the labels"
        )
    mode, p_mode = max(freqs.items(), key=lambda kv: kv[1])
    details = {"num_classes": k,
               "observed_classes": observed,
               "constant_policy_floor": 2 * p_mode / (1 + p_mode) / k,
               "mode": mode,
               "bound_direction": _MACRO_F1_UPPER}
    if k > observed:
        details["unsupported_classes"] = k - observed
        details["note"] = (
            f"{k - observed} declared class(es) have no support in this split and "
            "contribute F1 = 0; the floor is 1/K over the declared set."
        )
    return FloorResult(
        metric="macro_f1",
        floor=1.0 / k,
        policy="i.i.d. draws matching the label marginals (q = p)",
        band="B", label_free=False, bound="upper",
        details=details,
    )


# ---------------------------------------------------------------------------
# pass@k. Deterministic constant: p_mode, independent of k. Stochastic
# guesser drawing k i.i.d. answers from q: mean over items of
# 1 - (1 - q(y_i))^k, maximized by water-filling over answer frequencies.
# ---------------------------------------------------------------------------

def pass_at_k_floor(
    labels: Iterable | Mapping[object, int],
    k: int = 1,
    stochastic: bool = True,
) -> FloorResult:
    if k < 1:
        raise ValueError("k must be >= 1")
    freqs = _freqs(labels)

    mode, p_mode = max(freqs.items(), key=lambda kv: kv[1])
    if k == 1 or not stochastic:
        # With one draw the objective is linear in q: mode is optimal.
        return FloorResult(
            metric=f"pass@{k}",
            floor=p_mode,
            policy=f"constant modal answer {mode!r} (deterministic)",
            band="A", label_free=False,
            details={"k": k, "stochastic": False},
        )

    q = optimal_guess_distribution(freqs, k)
    floor = sum(p * (1 - (1 - q[a]) ** k) for a, p in freqs.items())
    support = {a: qa for a, qa in q.items() if qa > 0}
    return FloorResult(
        metric=f"pass@{k}",
        floor=floor,
        policy=f"k i.i.d. draws from water-filling distribution over "
               f"{len(support)} most frequent answers",
        band="A", label_free=False,
        details={"k": k, "stochastic": True, "support_size": len(support),
                 "deterministic_floor": p_mode},
    )


def pass_hat_k_floor(
    labels: Iterable | Mapping[object, int],
    k: int = 1,
) -> FloorResult:
    """pass^k: item counts as solved only if ALL k trials succeed.

    Distinct from pass@k (at least one of k). A stochastic policy succeeds on
    item i with probability q(y_i)^k, an objective convex in q, so its maximum
    over the simplex sits at a vertex -- the modal answer. The floor is
    p_mode for every k: flat, where the pass@k floor grows with k.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    freqs = _freqs(labels)
    mode, p_mode = max(freqs.items(), key=lambda kv: kv[1])
    return FloorResult(
        metric=f"pass^{k}",
        floor=p_mode,
        policy=f"constant modal answer {mode!r} (randomizing is strictly worse for k > 1)",
        band="A", label_free=False,
        details={"k": k, "flat_in_k": True, "mode": mode},
    )


def optimal_guess_distribution(freqs: Mapping[object, float], k: int) -> dict:
    """Maximize sum_a p_a * (1 - (1 - q_a)^k) over the simplex.

    Stationarity: p_a * k * (1 - q_a)^(k-1) = lam on the active set, so
    q_a = 1 - (lam / (k * p_a))^(1/(k-1)), clipped at 0. Solve for lam by
    bisection on the simplex constraint; sum_a q_a(lam) is decreasing in lam.
    """
    if k == 1:
        mode = max(freqs, key=freqs.get)
        return {a: (1.0 if a == mode else 0.0) for a in freqs}

    def q_of(lam: float) -> dict:
        out = {}
        for a, p in freqs.items():
            ratio = lam / (k * p)
            out[a] = max(0.0, 1.0 - ratio ** (1.0 / (k - 1)))
        return out

    lo, hi = 0.0, k * max(freqs.values())  # q(lo) sums to |A| >= 1; q(hi) sums to 0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(q_of(mid).values()) > 1.0:
            lo = mid
        else:
            hi = mid
    return q_of((lo + hi) / 2)


# ---------------------------------------------------------------------------
# Immune family: every input-oblivious policy scores at chance, in
# expectation. Nothing to compute; these exist so reports can say so.
# ---------------------------------------------------------------------------

def balanced_accuracy_floor(num_classes: int = 2) -> FloorResult:
    """Balanced accuracy = macro-averaged recall; floor 1/K for K classes.

    Binary balanced accuracy is the K = 2 case, giving the familiar 1/2. For
    K > 2 an input-oblivious policy drawing class c with probability q_c has
    recall_c = q_c, so the macro-average is (1/K) * sum_c q_c = 1/K exactly --
    immune for every K, not only for two classes.
    """
    if num_classes < 2:
        raise ValueError("balanced accuracy needs at least two classes")
    return FloorResult(
        metric="balanced_accuracy", floor=1.0 / num_classes, immune=True,
        band="B", label_free=True,
        policy="any input-oblivious policy (per-class recall = q_c, so the "
               "macro-average is 1/K by independence)",
        details={"num_classes": num_classes},
    )


def cohens_kappa_floor() -> FloorResult:
    """Cohen's kappa is 0 for any input-oblivious policy and any class count.

    The independence argument does not depend on K: observed agreement
    sum_c q_c p_c equals the chance agreement computed from the same marginals.
    """
    return FloorResult(
        metric="cohens_kappa", floor=0.0, immune=True,
        band="B", label_free=True, bound="expectation",
        policy="any input-oblivious policy (observed agreement = chance "
               "agreement, for any number of classes)",
        details={"bound_direction":
                 "exact on every realization for a constant policy; for a "
                 "randomized i.i.d. policy the 0 is an expectation, up to the "
                 "O(1/n) bias of a ratio with a policy-dependent denominator",
                 "concentration": "governed by the smallest per-class count, "
                                  "with a 1/(1 - p_e) factor that grows under "
                                  "extreme imbalance"},
    )


def mcc_floor() -> FloorResult:
    return FloorResult(
        metric="mcc", floor=0.0, immune=True,
        band="B", label_free=True, bound="expectation",
        policy="any input-oblivious policy (zero prediction-label covariance; "
               "0 by convention for constant predictions)",
        details={"bound_direction":
                 "exact on every realization for a constant policy; for a "
                 "randomized i.i.d. policy the 0 is an expectation, up to the "
                 "O(1/n) bias of a ratio with a policy-dependent denominator"},
    )


def roc_auc_floor() -> FloorResult:
    return FloorResult(
        metric="roc_auc", floor=0.5, immune=True,
        band="B", label_free=True,
        policy="any input-oblivious scorer (positive-negative score pairs "
               "exchangeable; constant scores tie at 1/2)",
    )


IMMUNE_METRICS = {
    "balanced_accuracy": balanced_accuracy_floor,
    "cohens_kappa": cohens_kappa_floor,
    "mcc": mcc_floor,
    "roc_auc": roc_auc_floor,
}


# ---------------------------------------------------------------------------
# Reporting helpers.
# ---------------------------------------------------------------------------

def capture_ratio(floor: float, sota: float) -> float:
    if sota <= 0:
        raise ValueError("SOTA score must be positive")
    return floor / sota


def headroom(floor: float, max_score: float = 1.0) -> float:
    """What a real agent can still demonstrate: max - floor.

    Unlike the capture ratio this needs no ceiling and no re-dating when the
    frontier moves, which is why the floor report leads with it.
    """
    if floor > max_score:
        raise ValueError("floor cannot exceed max_score")
    return max_score - floor


def normalized_score(score: float, floor: float,
                     max_score: float = 1.0) -> float | None:
    """Headroom-normalized score: (s - floor) / (max - floor).

    Returns None when the headroom is zero, i.e. when a degenerate policy
    already attains the metric's maximum. That is not a normalization failure
    but a terminal diagnosis: the pair concedes the whole scale, so there is
    nothing left to normalize against and the metric should be replaced rather
    than rescaled.
    """
    if floor > max_score:
        raise ValueError("floor cannot exceed max_score")
    gap = max_score - floor
    if gap == 0:
        return None
    return (score - floor) / gap
