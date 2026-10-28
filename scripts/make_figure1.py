"""Figure 1: dumbbell plot of degenerate floor vs. SOTA per benchmark-metric pair.

Each row is one audited pair on the absolute score axis (all metrics are rates
in [0,1]): open dot = best degenerate floor, filled dot = SOTA, connecting
segment = the headroom a real agent can demonstrate. Rows sorted by headroom
(1 - floor, annotated right), which every row has, rather than by capture
ratio, which needs a ceiling; pairs whose floor is the metric's maximum are
drawn in the warm accent. Rows with no verified same-split ceiling are drawn
floor-only, matching the "---" cells of Table 2.

Reads results/analytic_floors.json + results/executed_floors.json; pairs need
both floor and sota. --floors previews raw floors as a bar-less dot column
until SOTA is filled.

Run: .venv/bin/python scripts/make_figure1.py [--floors]
Output: paper/fig1_capture_ratio.pdf (vector, single column width)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# NeurIPS requires Type 1 or embedded TrueType only; matplotlib's PDF default
# is Type 3, so ship TrueType glyphs instead.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

ROOT = Path(__file__).resolve().parent.parent

# The audited panel: 18 benchmark-metric pairs (Table 2 aggregates three of
# them into single rows, so the table shows 14).
EXPECTED_PAIRS = 18

BLUE_DARK = "#1D4F8C"   # SOTA marker: one hue, dark shade
BLUE_EDGE = "#2A6DB0"   # floor marker edge: same hue, mid shade
HEADROOM = "#B9C6D8"    # connecting segment: same hue, light shade
ACCENT = "#C2452D"      # floor >= SOTA rows only (exception, not a series)
INK = "#1F2937"
INK_MUTED = "#6B7280"
GRID = "#E8EAEE"
BAND = "#F4F5F7"        # alternate-row shading: lighter than the grid

# Offset, in points from the axes' right edge, of the right-aligned headroom
# column. Wide enough that the column clears the "1.0" tick label.
HEADROOM_PAD = 42

# Marker diameter in points. The row pitch on the printed page is about 10pt,
# so anything larger than this makes neighbouring floors look like a column.
MARKER = 6.0


def load_entries() -> list[dict]:
    entries = []
    for name in ("analytic_floors.json", "executed_floors.json"):
        path = ROOT / "results" / name
        if path.exists():
            entries.extend(json.load(open(path)))
    entries = [e for e in entries if "error" not in e and e.get("floor") is not None]
    # GSM8K stores its stochastic pass@8 floor as a field; surface it as its
    # own pair so the k-scaling of the floor is visible in the figure.
    for e in list(entries):
        p8 = e.get("pass_at_8_stochastic_floor")
        if p8 is not None:
            entries.append({**e, "metric": "pass@8", "floor": p8,
                            "policy": "stochastic water-filling guesser"})
    return entries


# Row labels are spelled exactly as in Tables 2--3 of the paper (audit
# results and panel): tau as the
# Greek letter, initialisms upper-cased, policy names matching the table's.
BENCH_LABEL = {
    "GPQA (diamond)": "GPQA diamond",
    "BFCL (irrelevance)": "BFCL v4 irrelevance",
    "BFCL (live irrelevance)": "BFCL v4 live irrelevance",
    "BFCL (simple python)": "BFCL v4 (simple python)",
    "BFCL (live simple)": "BFCL v4 (live simple)",
    "BFCL (live multiple)": "BFCL v4 (live multiple)",
    "ALFWorld (fixed)": "ALFWorld (fixed legal action)",
    "ALFWorld (random)": "ALFWorld (random legal action)",
    "tau-bench airline (transfer)": "τ-bench airline (constant transfer)",
    "tau-bench airline (stop)": "τ-bench airline (constant stop)",
    "tau-bench retail (transfer)": "τ-bench retail (constant transfer)",
    "tau-bench retail (stop)": "τ-bench retail (constant stop)",
}

METRIC_LABEL = {
    "mcq_accuracy": "MCQ accuracy",
    "ast_accuracy": "AST accuracy",
    "irrelevance_accuracy": "abstention accuracy",
    "exact_match": "exact match",
    "success_rate": "success rate",
    "resolved_rate": "% resolved",
    "pass_hat_1": "pass$^1$",
}


def label(e: dict) -> str:
    bench = BENCH_LABEL.get(e["benchmark"], e["benchmark"])
    metric = METRIC_LABEL.get(e["metric"], e["metric"].replace("_", " "))
    return f"{bench}  ({metric})"


# Rows with no ceiling we could tie to a verified, same-split, primary source.
# Earlier drafts drew these as diamonds on dashed segments against an author
# estimate. The estimate could not change any conclusion -- these floors are at
# most 0.042 (GAIA), except BFCL irrelevance where the floor is the metric's
# maximum and the inversion is structural -- which is the reason not to draw it.
# They are plotted floor-only, matching the "---" cells in Table 2.
SOTA_NOT_PRINTED = {
    "BFCL (simple python)", "BFCL (live simple)", "BFCL (live multiple)",
    "BFCL (irrelevance)", "BFCL (live irrelevance)",
    "GAIA", "ALFWorld (fixed)", "ALFWorld (random)",
    # The constant-stop rows are a harness artifact (a lone canned response
    # never terminates the episode), so a ceiling on them would invite a
    # comparison the number does not support.
    "tau-bench retail (stop)", "tau-bench airline (stop)",
}


def sota_is_attributable(e: dict) -> bool:
    return e["benchmark"] not in SOTA_NOT_PRINTED


def band_rows(fig, ax, n: int, value_labels) -> None:
    """Shade alternate rows from the row label across to the headroom column.

    A row's label and its headroom number sit at opposite ends of a span that
    is mostly empty -- the floor-only rows draw a single dot and nothing else
    -- so the eye slips between rows on the way across. The band is the guide;
    it is drawn after tight_layout, when the label extents are final, and runs
    exactly to the text on both sides rather than to a guessed margin.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_axes = ax.transAxes.inverted()

    left = min(t.get_window_extent(renderer).x0 for t in ax.get_yticklabels())
    right = max(t.get_window_extent(renderer).x1 for t in value_labels)
    pad = abs(to_axes.transform((3, 0))[0] - to_axes.transform((0, 0))[0])
    x0 = to_axes.transform((left, 0))[0] - pad
    x1 = to_axes.transform((right, 0))[0] + pad

    trans = blended_transform_factory(ax.transAxes, ax.transData)
    for yi in range(0, n, 2):
        ax.add_patch(Rectangle((x0, yi - 0.5), x1 - x0, 1.0, transform=trans,
                               facecolor=BAND, edgecolor="none", zorder=-2,
                               clip_on=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--floors", action="store_true",
                        help="preview raw floors before SOTA is filled")
    args = parser.parse_args()

    entries = [e for e in load_entries() if args.floors or e.get("sota")]
    if not entries:
        raise SystemExit("nothing to plot: fill 'sota' fields in results/*.json "
                         "or run with --floors")
    # Sorted by headroom (1 - floor), which every row has, rather than by
    # capture ratio, which needs a ceiling and moves when models move. The two
    # orderings coincide on this panel; the difference is what the figure is
    # claiming to rank by.
    entries.sort(key=lambda e: e["floor"])  # largest floor ends up at the top row

    n = len(entries)
    # The audited panel is 18 benchmark-metric pairs. A short input file --
    # e.g. a partial analytic_floors.json -- would otherwise draw a smaller
    # figure that looks perfectly fine, so say so rather than plotting quietly.
    if n != EXPECTED_PAIRS:
        print(f"[warn] plotting {n} pairs, expected {EXPECTED_PAIRS}. Check that "
              "analytic_floors.json and executed_floors.json are both complete "
              "and that collect_results.py ran after analytic_floors.py.")

    # Canvas width is set just above the four-column legend's own width, so
    # the tight bbox is decided by the plot rather than by the legend; the
    # figure is then placed at full \linewidth, which is what keeps the type
    # legible. 18 rows at 0.30in each pushed it past half a page.
    fig, ax = plt.subplots(figsize=(6.8, 0.170 * n + 0.52))
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID, linewidth=0.7)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)

    value_labels = []
    for yi, e in enumerate(entries):
        floor = e["floor"]
        sota = e.get("sota") if sota_is_attributable(e) else None
        headroom = 1.0 - floor
        # Warm where the metric concedes everything: a floor at its maximum
        # ties or beats every imperfect model with no ceiling involved.
        inverted = headroom <= 0.0 or (sota is not None and floor >= sota)
        seg = ACCENT if inverted else HEADROOM
        edge = ACCENT if inverted else BLUE_EDGE

        if sota is not None:
            lo, hi = sorted((floor, sota))
            ax.plot([lo, hi], [yi, yi], color=seg, linewidth=1.8,
                    solid_capstyle="round", zorder=1, clip_on=False)
            ax.plot(sota, yi, "o", markersize=MARKER, color=BLUE_DARK, zorder=3,
                    clip_on=False)
        # floor: open marker (shape, not just color, separates it from SOTA).
        # clip_on=False everywhere: a floor of exactly 1.0 sits on the axes
        # edge, and the default clip cut those open circles in half.
        ax.plot(floor, yi, "o", markersize=MARKER, markerfacecolor="white",
                markeredgecolor=edge, markeredgewidth=1.5, zorder=4,
                clip_on=False)

        # Right-aligned so the decimal points form a column, and set far
        # enough out that the widest tick label ("1.0") has clear air.
        value_labels.append(
            ax.annotate(f"{headroom:.2f}", xy=(1.0, yi),
                        xycoords=("axes fraction", "data"),
                        xytext=(HEADROOM_PAD, 0), textcoords="offset points",
                        va="center", ha="right", fontsize=8,
                        color=ACCENT if inverted else INK_MUTED,
                        fontweight="bold" if inverted else "normal"))

    # A full row of air below the header: at 0.55 it crowded the first
    # value, which is the bold red 0.00 and the least forgiving row to crowd.
    ax.annotate("headroom", xy=(1.0, n - 1 + 0.95),
                xycoords=("axes fraction", "data"),
                xytext=(HEADROOM_PAD, 0), textcoords="offset points",
                va="center", ha="right", fontsize=8, color=INK_MUTED,
                style="italic")

    ax.set_yticks(range(n), [label(e) for e in entries], fontsize=8.5, color=INK)
    ax.set_xlabel("Score" if not args.floors else "Degenerate-baseline floor",
                  fontsize=8, color=INK)
    ax.tick_params(axis="x", labelsize=8, colors=INK)
    ax.tick_params(axis="y", length=0)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlim(-0.035, 1.035)
    ax.set_ylim(-0.8, n - 1 + 1.05)

    handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=MARKER,
               markerfacecolor="white", markeredgecolor=BLUE_EDGE,
               markeredgewidth=1.5, label="degenerate floor"),
        Line2D([], [], marker="o", linestyle="none", markersize=MARKER,
               color=BLUE_DARK, label="SOTA"),
        Line2D([], [], color=HEADROOM, linewidth=1.8, label="headroom"),
        Line2D([], [], marker="o", linestyle="none", markersize=MARKER,
               markerfacecolor="white", markeredgecolor=ACCENT,
               markeredgewidth=1.5, label="floor $=$ metric maximum"),
    ]
    # One row, four columns, parked in the blank band the y-limit leaves at
    # the top of the axes rather than above the figure: sitting outside, it
    # added its own height to a figure already fighting for page space, and
    # anchoring it to the axes pushed it off the right (the row labels make
    # the axes much narrower than the legend), so it is centred on the figure.
    legend = fig.legend(handles=handles, loc="lower center",
                        bbox_to_anchor=(0.5, 1.0),
                        bbox_transform=fig.transFigure, ncols=4, frameon=False,
                        fontsize=7.5, handletextpad=0.3, columnspacing=1.6,
                        borderaxespad=0.0)

    fig.tight_layout()
    # Re-anchor once the axes have their final position: the legend belongs
    # just above the top row, not at the top of the figure, and only
    # tight_layout knows where the axes ended up.
    legend.set_bbox_to_anchor((0.5, ax.get_position().y1 + 0.008),
                              transform=fig.transFigure)
    band_rows(fig, ax, n, value_labels)

    out = ROOT / "paper" / "fig1_capture_ratio.pdf"
    # legend and the CR column sit outside the axes; tight bbox keeps both.
    # CreationDate=None drops the embedded timestamp so a rerun on unchanged
    # results is byte-identical and git sees no new blob.
    fig.savefig(out, bbox_inches="tight", metadata={"CreationDate": None})
    print(f"wrote {out} ({n} pairs)")


if __name__ == "__main__":
    main()
