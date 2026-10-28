# floorcheck

Closed-form degenerate-baseline floors for benchmark metrics: what does a
policy that never reads its input score?

No dependencies. `pip install -e .` is all you need to recompute any floor in
the paper's Table 1. The test extra pulls in `pytest` and nothing else.

```bash
pip install -e .
pip install -e '.[test]' && pytest tests/
```

## Label-based floors

Labels are a JSON array `["A","B",...]` or a counts object `{"A": 120, ...}`.
The commands below run as written against the two files in `examples/`.

```bash
# accuracy / MCQ accuracy / exact match: floor is the modal-label rate
floorcheck compute examples/labels.json --metric mcq_accuracy --sota 0.896

# stochastic pass@k floor (water-filling guesser); grows with k
floorcheck compute examples/answers.json --metric pass@k --k 8

# pass^k floor (all k trials must pass); flat in k
floorcheck compute examples/answers.json --metric 'pass^k' --k 8

# macro-F1 floor (1/K via marginal matching, reported as `bound: upper`, since
# the draw's expected score only approaches 1/K). Pass --num-classes when the
# declared label set is larger than the classes present in this split,
# as it is for examples/labels.json: 10 options declared, 8 realized.
floorcheck compute examples/labels.json --metric macro_f1 --num-classes 10
```

## Rate-based and immune floors

```bash
# binary metrics need only the positive rate
floorcheck compute --metric binary_f1 --pi 0.35 --sota 0.7

# immune metrics: balanced accuracy is 1/K, so it takes --num-classes too
floorcheck compute --metric balanced_accuracy --num-classes 10
```

## Reporting fields

Every result carries `band`, `label_free` and `headroom` — checklist items 1,
2 and 5. Where the floor equals the metric's maximum, `normalized_score` is
`null` with a note naming the terminal diagnosis, rather than raising on the
division by zero.

```bash
# item 5: headroom is always emitted; --score adds the headroom-normalized
# (s - floor) / (1 - floor)
floorcheck compute examples/labels.json --metric accuracy --score 0.82

# item 6: a capture ratio is a dated claim about one ceiling, so give
# low:high where the ceiling is contested, as tau-bench airline's is
floorcheck compute --metric precision --pi 0.38 --sota 0.46:0.70 \
    --sota-source "frozen board / HAL / aggregator" --sota-as-of 2024-10

# everything applicable at once. A ceiling and a score belong to one metric,
# so report will not spread them across rows without --for-metric
floorcheck report examples/labels.json --num-classes 10 \
    --sota 0.82 --for-metric accuracy
```

## Python API

`accuracy_floor`, `binary_f1_floor`, `precision_floor`, `recall_floor`,
`auprc_floor`, `macro_f1_floor`, `pass_at_k_floor`, `pass_hat_k_floor`,
`optimal_guess_distribution` (water-filling), `capture_ratio`, `headroom`,
`normalized_score`, plus the immune family (`balanced_accuracy_floor`,
`cohens_kappa_floor`, `mcc_floor`, `roc_auc_floor`) whose floor is chance for
every input-oblivious policy.

Invoke the tool as `floorcheck` rather than importing it from the repository
root, where the source directory shadows the installed package.

Apache 2.0 license.
