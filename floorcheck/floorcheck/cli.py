"""floorcheck CLI.

Usage:
  floorcheck compute labels.json --metric accuracy [--sota 0.82]
  floorcheck compute labels.json --metric pass@k --k 8 --sota 0.9
  floorcheck compute --pi 0.35 --metric binary_f1 --sota 0.7
  floorcheck report labels.json --sota 0.82   # all applicable label metrics

labels.json: either a JSON array of labels (["A","B","A",...]) or an object
of counts ({"A": 120, "B": 80}).
"""

from __future__ import annotations

import argparse
import json
import sys

from floorcheck import closed_forms as cf


def _load_labels(path: str):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, (list, dict)):
        return data
    raise SystemExit("labels file must be a JSON array of labels or object of counts")


def _parse_sota(raw: str | None) -> tuple[float, ...] | None:
    """A SOTA is one number, or `low:high` when the ceiling is contested.

    Checklist item 6 asks for a range wherever the ceiling is disputed, so the
    scalar form alone cannot express the report the paper asks for -- tau-bench
    airline needs 0.46:0.70.
    """
    if raw is None:
        return None
    try:
        parts = tuple(float(p) for p in raw.split(":"))
    except ValueError:
        raise SystemExit(f"--sota must be a number or low:high, got {raw!r}")
    if len(parts) not in (1, 2):
        raise SystemExit(f"--sota must be a number or low:high, got {raw!r}")
    if len(parts) == 2 and parts[0] > parts[1]:
        raise SystemExit("--sota range must be given low first")
    return parts


def _emit(result: cf.FloorResult, sota: tuple[float, ...] | None,
          score: float | None = None, source: str | None = None,
          as_of: str | None = None) -> None:
    out = {
        "metric": result.metric,
        "band": result.band,
        "floor": round(result.floor, 6),
        "headroom": round(cf.headroom(result.floor), 6),
        "policy": result.policy,
        "label_free": result.label_free,
        "immune": result.immune,
        "bound": result.bound,
        **({"details": result.details} if result.details else {}),
    }
    if score is not None:
        out["score"] = score
        norm = cf.normalized_score(score, result.floor)
        out["normalized_score"] = None if norm is None else round(norm, 6)
        if norm is None:
            out["normalized_score_note"] = (
                "floor equals the metric's maximum, so there is no headroom to "
                "normalize against. Replace the metric on this pair rather than "
                "rescaling scores against it."
            )
    if sota is not None:
        # Dividing by the larger ceiling gives the smaller ratio, so the range
        # comes out reversed relative to the SOTA range it came from.
        ratios = sorted(round(cf.capture_ratio(result.floor, s), 6) for s in sota)
        out["sota"] = sota[0] if len(sota) == 1 else list(sota)
        if len(sota) == 1:
            out["capture_ratio"] = ratios[0]
        else:
            out["capture_ratio_range"] = ratios
        # Item 6 is a dated claim, not just a number; carry the provenance the
        # caller supplied rather than emitting a bare ratio.
        out["sota_source"] = source
        out["sota_as_of"] = as_of
    json.dump(out, sys.stdout, indent=2)
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="floorcheck", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_compute = sub.add_parser("compute", help="floor for one metric")
    p_compute.add_argument("labels", nargs="?", help="labels JSON (array or counts object)")
    p_compute.add_argument("--metric", required=True,
                           choices=["accuracy", "mcq_accuracy", "exact_match", "pass@k",
                                    "pass^k", "macro_f1", "binary_f1", "precision",
                                    "recall", "auprc", "balanced_accuracy",
                                    "cohens_kappa", "mcc", "roc_auc"])
    p_compute.add_argument("--pi", type=float, help="positive rate (binary metrics)")
    p_compute.add_argument("--k", type=int, default=1, help="k for pass@k / pass^k")
    p_compute.add_argument("--num-classes", type=int, dest="num_classes",
                           help="declared class count: macro_f1 (if larger than "
                                "the classes present in the label file) and "
                                "balanced_accuracy (floor 1/K; default binary)")
    p_compute.add_argument("--deterministic", action="store_true",
                           help="pass@k: constant answer instead of optimal stochastic guesser")
    p_compute.add_argument("--sota", help="SOTA on this same split and metric, "
                                          "as a number or low:high when contested")
    p_compute.add_argument("--sota-source", dest="sota_source",
                           help="where the SOTA came from (checklist item 6)")
    p_compute.add_argument("--sota-as-of", dest="sota_as_of",
                           help="the date the SOTA is current to, which is not "
                                "the date you retrieved it (checklist item 6)")
    p_compute.add_argument("--score", type=float,
                           help="a model's score, for the headroom-normalized "
                                "(s - floor)/(1 - floor)")

    p_report = sub.add_parser("report", help="all label-based floors at once")
    p_report.add_argument("labels")
    p_report.add_argument("--sota", help="see `compute --sota`; requires "
                                         "--for-metric to say which row it is on")
    p_report.add_argument("--sota-source", dest="sota_source")
    p_report.add_argument("--sota-as-of", dest="sota_as_of")
    p_report.add_argument("--score", type=float,
                          help="a model's score, for the headroom-normalized "
                               "(s - floor)/(1 - floor); requires --for-metric")
    p_report.add_argument("--for-metric", dest="for_metric",
                          help="which emitted row --sota / --score belong to "
                               "(e.g. accuracy, macro_f1, pass@8). A ceiling or "
                               "a score is defined on one metric; this report "
                               "spans several, so it will not broadcast them.")
    p_report.add_argument("--num-classes", type=int, dest="num_classes",
                          help="declared class count, used for both the macro_f1 "
                               "and balanced_accuracy floors (defaults to the "
                               "classes present in the label file)")
    p_report.add_argument("--k", type=int, nargs="*", default=[1, 8],
                          help="k values for pass@k rows")

    args = parser.parse_args(argv)
    sota = _parse_sota(args.sota)

    if args.cmd == "compute":
        m = args.metric
        if m in cf.IMMUNE_METRICS:
            if m == "balanced_accuracy":
                result = cf.balanced_accuracy_floor(args.num_classes or 2)
            else:
                result = cf.IMMUNE_METRICS[m]()
        elif m in ("binary_f1", "precision", "recall", "auprc"):
            if args.pi is None:
                parser.error(f"--pi required for {m}")
            result = {"binary_f1": cf.binary_f1_floor, "precision": cf.precision_floor,
                      "recall": cf.recall_floor, "auprc": cf.auprc_floor}[m](args.pi)
        elif m == "macro_f1" and args.labels is None:
            # Band B: the floor is 1/K over the declared class set, so the
            # table's "no label file needed" has to hold at the CLI too.
            if args.num_classes is None:
                parser.error("macro_f1 needs a labels file or --num-classes")
            result = cf.macro_f1_floor(num_classes=args.num_classes)
        else:
            if args.labels is None:
                parser.error(f"labels file required for {m}")
            labels = _load_labels(args.labels)
            if m == "pass@k":
                result = cf.pass_at_k_floor(labels, k=args.k,
                                            stochastic=not args.deterministic)
            elif m == "pass^k":
                result = cf.pass_hat_k_floor(labels, k=args.k)
            elif m == "macro_f1":
                result = cf.macro_f1_floor(labels, num_classes=args.num_classes)
            else:
                result = {"accuracy": cf.accuracy_floor,
                          "mcq_accuracy": cf.mcq_accuracy_floor,
                          "exact_match": cf.exact_match_floor}[m](labels)
        _emit(result, sota, args.score, args.sota_source, args.sota_as_of)

    elif args.cmd == "report":
        # A SOTA and a score each live on one metric (Definition 2: same split
        # *and* metric). This subcommand emits several metrics, so broadcasting
        # one of them across all rows would manufacture capture ratios the
        # paper's own definition forbids. Make the caller name the row.
        if (sota is not None or args.score is not None) and not args.for_metric:
            parser.error(
                "--sota / --score apply to one metric; pass --for-metric to say "
                "which emitted row they belong to (e.g. --for-metric accuracy). "
                "A capture ratio is only defined against a ceiling on the same "
                "split and metric."
            )

        def emit(result: cf.FloorResult) -> None:
            on_row = result.metric == args.for_metric
            _emit(result,
                  sota if on_row else None,
                  args.score if on_row else None,
                  args.sota_source if on_row else None,
                  args.sota_as_of if on_row else None)

        labels = _load_labels(args.labels)
        emitted = []
        emitted.append(cf.accuracy_floor(labels))
        if len(cf._freqs(labels)) >= 2:
            emitted.append(cf.macro_f1_floor(labels, num_classes=args.num_classes))
        for k in args.k:
            if k > 1:
                emitted.append(cf.pass_at_k_floor(labels, k=k))
                emitted.append(cf.pass_hat_k_floor(labels, k=k))
        n_classes = args.num_classes or len(cf._freqs(labels))
        for name, immune in cf.IMMUNE_METRICS.items():
            emitted.append(cf.balanced_accuracy_floor(n_classes)
                           if name == "balanced_accuracy" else immune())

        if args.for_metric and args.for_metric not in {r.metric for r in emitted}:
            parser.error(
                f"--for-metric {args.for_metric!r} is not one of the rows this "
                f"report emits: {', '.join(sorted(r.metric for r in emitted))}"
            )
        for result in emitted:
            emit(result)


if __name__ == "__main__":
    main()
