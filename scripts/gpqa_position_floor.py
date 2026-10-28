"""Realized-shuffle floor for a constant-position policy on GPQA diamond.

GPQA ships the correct answer in its own column and the three distractors in
three more (`Correct Answer`, `Incorrect Answer 1..3`); the option *order* a
model sees is produced by the evaluation harness, not by the dataset. So the
constant-position policy ("always answer C") does not score 1/K on a given
run: it scores the frequency with which the harness's shuffle happens to put
the correct option in that position, and the best such policy scores the
maximum of those four frequencies.

That maximum is upward-biased at finite n even under a perfect shuffle -- it
is a maximum over K correlated sample proportions -- which is the same
finite-sample selection effect Appendix A.4 documents for average precision.
This script measures the size of it by simulating uniform shuffles, so the
paper can state the realization band rather than assert 1/K.

Run: .venv/bin/python scripts/gpqa_position_floor.py [--reps 200000]

Writes results/gpqa_position_floor.json. No GPQA download is needed: only n
and K enter the simulation, and both are fixed by the split (n = 198, K = 4).
Pass --verify-n to check n against the gated HF copy when it is available.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"

N_DIAMOND = 198
K_OPTIONS = 4
SEED = 0


def simulate(n: int, k: int, reps: int, seed: int) -> dict:
    """Distribution of max_j p_hat_j over uniform option shuffles."""
    rng = random.Random(seed)
    maxima = []
    for _ in range(reps):
        counts = [0] * k
        for _ in range(n):
            counts[rng.randrange(k)] += 1
        maxima.append(max(counts) / n)
    maxima.sort()

    def pct(q: float) -> float:
        return maxima[min(len(maxima) - 1, int(q * len(maxima)))]

    mean = sum(maxima) / len(maxima)
    var = sum((m - mean) ** 2 for m in maxima) / (len(maxima) - 1)
    return {
        "mean": mean,
        "median": pct(0.50),
        "p05": pct(0.05),
        "p95": pct(0.95),
        "sd": var ** 0.5,
        "min": maxima[0],
        "max": maxima[-1],
        "excess_over_uniform": mean - 1.0 / k,
    }


def single_position_sd(n: int, k: int) -> float:
    """SD of one position's frequency: the band around 1/K for a *fixed* choice."""
    p = 1.0 / k
    return (p * (1 - p) / n) ** 0.5


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200_000)
    ap.add_argument("--n", type=int, default=N_DIAMOND)
    ap.add_argument("--k", type=int, default=K_OPTIONS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--verify-n", action="store_true",
                    help="check n against the gated HF copy of gpqa_diamond")
    args = ap.parse_args()

    n = args.n
    if args.verify_n:
        from datasets import load_dataset
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        if len(ds) != n:
            raise SystemExit(f"n mismatch: dataset has {len(ds)}, expected {n}")

    best = simulate(n, args.k, args.reps, args.seed)
    out = {
        "_note": (
            "Constant-position policy on a *realized* GPQA option shuffle. "
            "The reported floor of 0.250 is the expectation over shuffles; on "
            "any one shuffle the best constant-position policy scores "
            "max_j p_hat_j, whose mean is the 'best_position' block below. "
            "Standard harnesses reshuffle per evaluation run, so the excess is "
            "not exploitable ex ante -- it is a realization band, not a higher "
            "floor. Reported so the band is stated rather than assumed away."
        ),
        "n": n,
        "k": args.k,
        "reps": args.reps,
        "seed": args.seed,
        "uniform_floor": 1.0 / args.k,
        "best_position": best,
        "fixed_position_sd": single_position_sd(n, args.k),
    }
    path = RESULTS / "gpqa_position_floor.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
