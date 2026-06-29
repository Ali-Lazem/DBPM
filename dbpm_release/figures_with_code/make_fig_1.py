#!/usr/bin/env python3
"""
make_fig1_standalone.py
Standalone renderer for the redesigned Figure 4 (five candidate signals).
Run:  python3 make_fig1_standalone.py --out-prefix fig_1
Drop-in: the fig1_iterations() body below can replace the same-named
function in make_paper_figures_v6.py (same deps: PALETTE, _style_axes, _save).
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {
    "selective": "#0072B2", "marginal": "#E69F00", "anti": "#D55E00",
    "absent": "#999999", "neutral": "#4A4A4A", "axis": "#2B2B2B", "grid": "#D1D1D1",
}
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": PALETTE["axis"], "axes.labelcolor": PALETTE["axis"],
    "axes.titlecolor": PALETTE["axis"], "axes.titleweight": "bold",
    "text.color": PALETTE["axis"], "xtick.color": PALETTE["axis"],
    "ytick.color": PALETTE["axis"],
})
FONT_TITLE = 11.5; FONT_AX = 9; FONT_TICK = 8.5; FONT_ANNOT = 8


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


def fig1_iterations(out_prefix):
    # (name, lift, failure_mode_label, is_selective)
    signals = [
        ("Category-aggregate fail-mass", 0.006, "aggregation-level mismatch", False),
        ("Verifier uncertainty",          None, "signal absence (flag unpopulated)", False),
        ("Per-category fail-rate",        0.05, "counting asymmetry", False),
        ("Per-patient difficulty",        0.00, "cross-task orthogonality", False),
        ("NER-coverage intersection",     1.84, "structurally aligned (selective)", True),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.8), layout="constrained")

    ys = list(range(len(signals)))[::-1]  # listed order, top to bottom
    threshold = 1.5

    for y, (name, lift, mode, selective) in zip(ys, signals):
        color = PALETTE["selective"] if selective else PALETTE["anti"]
        val = 0.0 if lift is None else lift
        ax.barh(y, val, height=0.55, color=color, edgecolor="#222222",
                linewidth=0.8, zorder=3)

        if lift is None:
            vlabel = "no signal"
        elif lift < threshold:
            vlabel = f"lift {lift:.3f}" if lift < 0.1 else f"lift {lift:.2f}"
        else:
            vlabel = f"lift {lift:.2f}"
        outcome = "SELECTIVE" if selective else "not selective"
        ax.text(max(val, 0.0) + 0.04, y + 0.13,
                f"{vlabel}  ({outcome})", va="center", ha="left",
                fontsize=FONT_ANNOT, fontweight="bold",
                color=(PALETTE["selective"] if selective else PALETTE["anti"]))
        ax.text(max(val, 0.0) + 0.04, y - 0.17, mode, va="center", ha="left",
                fontsize=FONT_ANNOT - 0.5, style="italic", color=PALETTE["neutral"])

    # threshold line
    ax.axvline(threshold, color=PALETTE["selective"], linestyle="--",
               linewidth=1.3, alpha=0.85, zorder=2)
    # selective-zone shading
    ax.axvspan(threshold, 2.15, color=PALETTE["selective"], alpha=0.05, zorder=0)

    # --- FIX: place threshold label INSIDE the plot, mid-height, so it can
    # never collide with the title above. Anchored at the bottom (success) bar. ---
    ax.text(threshold + 0.03, 1.15, "selectivity\nthreshold (1.5)",
            fontsize=FONT_ANNOT, color=PALETTE["selective"],
            fontweight="bold", va="center", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in signals], fontsize=FONT_AX)
    ax.set_ylim(-0.6, len(signals) - 0.4)
    ax.set_xlim(0, 2.15)
    ax.set_xlabel("Selectivity lift  =  P(tag | FAIL) / P(tag | PASS)",
                  fontsize=FONT_AX, fontweight="bold")
    # single-line title, kept clear of the top-right threshold label
    ax.set_title("Five candidate signals for the question-answering gate",
                 fontsize=FONT_TITLE, pad=10)

    _style_axes(ax)
    ax.grid(False)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.6,
            linestyle=":", zorder=0)

    _save(fig, out_prefix)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-prefix", default="fig_1")
    args = ap.parse_args()
    fig1_iterations(args.out_prefix)
    print("done")
