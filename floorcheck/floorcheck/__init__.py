"""floorcheck: closed-form degenerate-baseline floors for benchmark metrics."""

from floorcheck.closed_forms import (
    FloorResult,
    IMMUNE_METRICS,
    accuracy_floor,
    auprc_floor,
    balanced_accuracy_floor,
    binary_f1_floor,
    capture_ratio,
    cohens_kappa_floor,
    exact_match_floor,
    headroom,
    macro_f1_floor,
    mcc_floor,
    mcq_accuracy_floor,
    normalized_score,
    optimal_guess_distribution,
    pass_at_k_floor,
    pass_hat_k_floor,
    precision_floor,
    recall_floor,
    roc_auc_floor,
)

__version__ = "0.1.0"

__all__ = [
    "FloorResult",
    "IMMUNE_METRICS",
    "accuracy_floor",
    "auprc_floor",
    "balanced_accuracy_floor",
    "binary_f1_floor",
    "capture_ratio",
    "cohens_kappa_floor",
    "exact_match_floor",
    "headroom",
    "macro_f1_floor",
    "mcc_floor",
    "mcq_accuracy_floor",
    "normalized_score",
    "optimal_guess_distribution",
    "pass_at_k_floor",
    "pass_hat_k_floor",
    "precision_floor",
    "recall_floor",
    "roc_auc_floor",
]
