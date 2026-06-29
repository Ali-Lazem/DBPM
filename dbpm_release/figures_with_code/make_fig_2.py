#!/usr/bin/env python3
"""
make_fig2_standalone.py
Redesigned Figure 5 (per-category operating envelope of the NER-coverage gate).
Fixes reviewer issues: log x-axis handles the 17.13 outlier honestly (no clamp,
no triangle); single marker type (bars, not dots+triangles), so no spurious
shape dimension; colour = regime only; categories grouped by regime; title makes
clear this is the ONE working signal's per-category behaviour, not the five-signal
comparison. Coverage is annotated as text per bar (a true 2nd quantity) rather
than smuggled into marker geometry.
Run: python3 make_fig2_standalone.py --out-prefix fig_2
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PALETTE = {
    "selective": "#0072B2", "marginal": "#E69F00", "categorical": "#CC79A7",
    "saturated": "#56B4E9", "fallback": "#999999", "axis": "#2B2B2B",
    "grid": "#D1D1D1", "neutral": "#4A4A4A",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": PALETTE["axis"], "axes.labelcolor": PALETTE["axis"],
    "axes.titlecolor": PALETTE["axis"], "axes.titleweight": "bold",
    "text.color": PALETTE["axis"], "xtick.color": PALETTE["axis"],
    "ytick.color": PALETTE["axis"],
})
FONT_TITLE = 11.5; FONT_AX = 9; FONT_TICK = 8.5; FONT_ANNOT = 7.8


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_TICK, width=0.8)


def _save(fig, out_path_no_ext):
    for ext in ("png", "pdf"):
        p = f"{out_path_no_ext}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    facecolor="white", transparent=False, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def fig2_envelope(out_prefix):
    # (category, lift, coverage%, regime)  -- lift None = no tagging (undefined)
    # grouped by regime, most-selective at top
    rows = [
        ("diagnosis",                17.13, 95.7, "selective"),
        ("risk_factor",               2.16, 53.7, "selective"),
        ("outcome_clinicalstatus",    1.05,  4.4, "categorical"),
        ("outcome_disposition",       1.02,  1.7, "categorical"),
        ("symptoms",                  None, 96.8, "saturated"),
        ("treatment",                 None, 97.7, "saturated"),
        ("tests",                     None, 94.8, "saturated"),
        ("history",                   None, 42.8, "saturated"),
        ("observation",               None,  1.3, "saturated"),
        ("adverse_event",             0.00,  0.0, "fallback"),
        ("outcome_mortality",         0.00,  0.0, "fallback"),
    ]

    # log axis floor for plotting null/zero lifts
    FLOOR = 0.04
    fig, ax = plt.subplots(figsize=(9.4, 5.4), layout="constrained")

    ys = list(range(len(rows)))[::-1]
    for y, (cat, lift, cov, regime) in zip(ys, rows):
        color = PALETTE[regime]
        if lift is None or lift == 0.0:
            xval = FLOOR  # tiny stub so the row is visible on log scale
            bar_alpha = 0.35
        else:
            xval = lift
            bar_alpha = 1.0
        ax.barh(y, xval, height=0.6, color=color, edgecolor="#222222",
                linewidth=0.7, alpha=bar_alpha, zorder=3)

        # value label
        if lift is None:
            vtxt = "no tagging"
        elif lift == 0.0:
            vtxt = "fallback (lift 0.00)"
        else:
            vtxt = f"lift {lift:.2f}"
        ax.text(xval * 1.12 if xval > FLOOR else FLOOR * 1.3, y,
                f"{vtxt}   cov {cov:.0f}%", va="center", ha="left",
                fontsize=FONT_ANNOT, color=PALETTE["axis"])

    # threshold
    ax.axvline(1.5, color=PALETTE["selective"], linestyle="--",
               linewidth=1.2, alpha=0.85, zorder=2)
    ax.axvline(1.0, color=PALETTE["neutral"], linestyle=":",
               linewidth=1.0, alpha=0.5, zorder=2)

    ax.set_xscale("log")
    ax.set_xlim(FLOOR * 0.8, 40)
    ax.set_xticks([0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 20.0])
    ax.set_xticklabels(["0.1", "0.5", "1.0", "1.5", "2.0", "5", "10", "20"])
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=FONT_AX, family="monospace")
    ax.set_ylim(-0.7, len(rows) - 0.3)

    ax.set_xlabel("Selectivity lift  =  P(tag | FAIL) / P(tag | PASS)   (log scale)",
                  fontsize=FONT_AX, fontweight="bold")
    fig.text(0.5, -0.04,
             "cov = coverage: fraction of PASS pairs whose patient NER set contains "
             "the asked category. Selectivity requires mid-range coverage.",
             ha="center", fontsize=FONT_ANNOT - 0.3, style="italic",
             color=PALETTE["neutral"])
    ax.set_title("Per-category behaviour of the NER-coverage gate (the working signal), "
                 "held-out cohort", fontsize=FONT_TITLE, pad=14)

    # threshold annotation
    ax.text(1.55, 4.5, "selectivity\nthreshold 1.5",
            fontsize=FONT_ANNOT, color=PALETTE["selective"],
            fontweight="bold", va="center", ha="left")

    _style_axes(ax)
    ax.grid(True, axis="x", which="both", color=PALETTE["grid"],
            linewidth=0.5, linestyle=":", zorder=0)

    legend_items = [
        ("selective (lift > 1.5)", PALETTE["selective"]),
        ("categorical (lift ~ 1.0, sparse coverage)", PALETTE["categorical"]),
        ("saturated / no tagging", PALETTE["saturated"]),
        ("difficulty-gate fallback (no NER bucket)", PALETTE["fallback"]),
    ]
    handles = [Patch(facecolor=c, edgecolor="#222222", linewidth=0.7, label=l)
               for l, c in legend_items]
    ax.legend(handles=handles, loc="lower right", fontsize=FONT_ANNOT,
              frameon=True, framealpha=0.97, edgecolor=PALETTE["grid"])

    _save(fig, out_prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="fig_2")
    args = ap.parse_args()
    fig2_envelope(args.out_prefix)
    print("done")
