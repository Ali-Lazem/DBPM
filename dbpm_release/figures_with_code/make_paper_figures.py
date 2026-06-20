#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
except ImportError as e:
    print(f"[FATAL] matplotlib required: {e}")
    sys.exit(1)


# -- Professional Publication Style ----------------------------------
PALETTE = {
    "selective":   "#0072B2",  # Deep scientific blue
    "marginal":    "#E69F00",  # Muted gold/orange
    "anti":        "#D55E00",  # Vermillion / Brick Red
    "absent":      "#999999",  # Neutral slate gray
    "categorical": "#CC79A7",  # Soft magenta/purple
    "saturated":   "#56B4E9",  # Sky blue
    "no_bucket":   "#E5E5E5",  # Light background gray
    "neutral":     "#4A4A4A",  # Charcoal
    "axis":        "#2B2B2B",  # Off-black for text readability
    "grid":        "#D1D1D1",  # Soft grid
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "axes.edgecolor": PALETTE["axis"],
    "axes.labelcolor": PALETTE["axis"],
    "axes.titlecolor": PALETTE["axis"],
    "axes.titleweight": "bold",
    "text.color": PALETTE["axis"],
    "xtick.color": PALETTE["axis"],
    "ytick.color": PALETTE["axis"],
})

FONT_TITLE = 11.5
FONT_AX = 9
FONT_TICK = 8.5
FONT_ANNOT = 8


def _style_axes(ax):
    """Applies a clean, minimal scientific aesthetic to axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", color=PALETTE["grid"],
            linewidth=0.6, linestyle=":", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=FONT_TICK, width=0.8)


def _save(fig, out_path_no_ext):
    for ext in ("png", "pdf"):
        p = f"{out_path_no_ext}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None,
                    facecolor="white", transparent=False, bbox_inches="tight")
        print(f"  wrote {p}")
    plt.close(fig)


def _load(p, optional=False):
    if not p.exists():
        if optional:
            return None
        print(f"[FATAL] missing report: {p}")
        sys.exit(2)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"[FATAL] JSON parse {p}: {e}")
        sys.exit(2)


# -----------------------------------------------------------------
# Figure 1: Five signal sources for the question-answering gate
# -----------------------------------------------------------------
def fig1_iterations(out_prefix):
    iterations = [
        ("category-aggregate\nfail-mass", 0.006, "anti",      "category-aggregate\ntags by volume"),
        ("verifier\nuncertainty",         None,  "absent",    "verdict_path flag\nunpopulated at scale"),
        ("per-category\nfail-rate",       0.05,  "anti",      "dual-bucket fail-rate\n-> counting asymmetry"),
        ("per-patient\ndifficulty",       0.00,  "absent",    "difficulty distributions\nindistinguishable"),
        ("NER-coverage\nintersection",    1.84,  "selective", "NER-coverage\nintersection (selective)"),
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.5, 6.0),
        gridspec_kw={"height_ratios": [1, 1.6], "hspace": 0.05},
        sharex=True, layout="constrained"
    )

    xs = list(range(len(iterations)))
    heights = [(0.0 if h is None else h) for _, h, _, _ in iterations]
    colors = [PALETTE[r if r in PALETTE else "absent"] for _, _, r, _ in iterations]

    for ax in (ax_top, ax_bot):
        ax.bar(xs, heights, color=colors, edgecolor="#222222", linewidth=0.8, zorder=2)
        _style_axes(ax)

    ax_top.set_ylim(1.5, 2.1)
    ax_bot.set_ylim(0, 0.35)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)

    d = 0.015
    kwargs = dict(transform=ax_top.transAxes, color=PALETTE["axis"], clip_on=False, linewidth=1.2)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bot.transAxes)
    ax_bot.plot((-d, +d), (1 - d, 1 + d), **kwargs)

    ax_top.axhline(1.5, color=PALETTE["selective"], linestyle="--", linewidth=1.2, alpha=0.8, zorder=1)
    ax_bot.axhline(0.0, color=PALETTE["axis"], linewidth=0.8, alpha=0.5)

    for x, (_, h, _, _) in zip(xs, iterations):
        if h is not None and h >= 1.5:
            ax_top.text(x, h + 0.03, f"{h:.2f}", ha="center", va="bottom",
                        fontsize=FONT_ANNOT+1, fontweight="bold", color=PALETTE["axis"])

    for x, (_, h, regime, _) in zip(xs, iterations):
        if h is None:
            ax_bot.text(x, 0.015, "no\nsignal", ha="center", va="bottom",
                        fontsize=FONT_ANNOT, style="italic", color=PALETTE["neutral"])
        elif h < 1.5:
            ax_bot.text(x, h + 0.015, f"{h:.3f}" if h < 0.1 else f"{h:.2f}",
                        ha="center", va="bottom", fontsize=FONT_ANNOT+1, fontweight="bold")

    combined_labels = [f"{title}\n\n{desc}" for title, _, _, desc in iterations]
    ax_bot.set_xticks(xs)
    ax_bot.set_xticklabels(combined_labels, fontsize=FONT_AX, linespacing=1.4)

    ax_top.text(-0.4, 1.55, "selectivity threshold (1.5)",
                fontsize=FONT_ANNOT, color=PALETTE["selective"], fontweight="bold")

    fig.supylabel("Selectivity lift  =  P(tag | FAIL) / P(tag | PASS)", fontsize=FONT_AX+1)
    fig.suptitle("Five signal sources for the question-answering gate: one structurally aligned signal",
                 fontsize=FONT_TITLE)

    _save(fig, out_prefix)


# -----------------------------------------------------------------
# Figure 2: per-category operating envelope (dot chart)
# -----------------------------------------------------------------
def fig2_envelope(report, out_prefix):
    if report is None: return
    per_cat = report.get("per_category", [])
    if not per_cat: return

    def _coverage_pct(r):
        cov_pass = r.get("pass_with_ner_coverage", 0) or 0
        nocov_pass = r.get("pass_without_ner_coverage", 0) or 0
        denom = cov_pass + nocov_pass
        return 100.0 * cov_pass / denom if denom > 0 else 0.0

    def _regime(r, cov_pct):
        if r.get("no_ner_bucket"): return "no_bucket"
        lift = r.get("lift")
        if lift is None: return "saturated"
        if isinstance(lift, str) or lift >= 1.5: return "selective"
        if 0.95 <= lift < 1.5: return "categorical" if cov_pct < 10 else "marginal"
        return "anti"

    def _sort_key(r):
        lift = r.get("lift")
        if lift is None: return -2
        if isinstance(lift, str): return float("inf")
        return lift

    per_cat_sorted = sorted(per_cat, key=_sort_key, reverse=True)

    fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    ys = list(range(len(per_cat_sorted)))

    max_finite = max([r.get("lift") for r in per_cat_sorted if isinstance(r.get("lift"), (int, float))] + [1.5])
    is_clipped = max_finite > 4.5
    clip_x = 3.8 # The visual cutoff point for extreme outliers

    for y, r in zip(ys, per_cat_sorted):
        cov = _coverage_pct(r)
        regime = _regime(r, cov)
        color = PALETTE[regime]
        lift = r.get("lift")

        marker = "o"

        if lift is None:
            x, label = 0.0, f"no tagging  -  cov {cov:.0f}%"
        elif isinstance(lift, str):
            x, label = 3.2, f"lift = inf  -  cov {cov:.0f}%"
        else:
            label = f"lift {lift:.2f}  -  cov {cov:.0f}%"
            # [FIXED PROPORTIONS]: Use an arrow marker for massive outliers
            if is_clipped and lift > clip_x:
                x = clip_x
                marker = ">"
            else:
                x = lift

        ax.scatter([x], [y], s=160, color=color, edgecolors="#222222",
                   linewidths=0.8, marker=marker, zorder=3)
        ax.text(x + 0.09, y, label, va="center", fontsize=FONT_ANNOT, color=PALETTE["axis"])

    ax.axvline(1.0, color=PALETTE["neutral"], linewidth=1.0, alpha=0.4, zorder=1)
    ax.axvline(1.5, color=PALETTE["selective"], linewidth=1.2, linestyle="--", alpha=0.8, zorder=1)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["category"] for r in per_cat_sorted], fontsize=FONT_AX)
    ax.invert_yaxis()

    # Create enough right-side buffer so the "lift 17.13" text fits, but don't stretch the X-axis to 17
    if is_clipped:
        ax.set_xlim(-0.1, clip_x + 1.2)
    else:
        ax.set_xlim(-0.1, max_finite + 0.8)

    ax.set_xlabel("Selectivity lift on held-out cohort (P(tag|FAIL) / P(tag|PASS))", fontsize=FONT_AX, fontweight="bold")
    ax.set_title("NER-coverage gate: per-category operating envelope (DBPM-full, held-out cohort)", fontsize=FONT_TITLE)
    _style_axes(ax)
    ax.grid(False)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.6, linestyle=":", zorder=0)

    legend_items = [
        ("selective (lift ≥ 1.5)", PALETTE["selective"]),
        ("marginal (1.0 ≤ lift < 1.5)", PALETTE["marginal"]),
        ("anti-selective (lift < 1.0)", PALETTE["anti"]),
        ("categorical / sparse cov", PALETTE["categorical"]),
        ("saturated (no tagging)", PALETTE["saturated"]),
        ("no-NER-bucket", PALETTE["no_bucket"]),
    ]
    handles = [Patch(facecolor=c, edgecolor="#222222", linewidth=0.8, label=lbl) for lbl, c in legend_items]
    ax.legend(handles=handles, loc="lower right", fontsize=FONT_ANNOT, frameon=True,
              framealpha=0.98, edgecolor=PALETTE["grid"])

    _save(fig, out_prefix)


# -----------------------------------------------------------------
# Figure 3: Production-scale verifier-fed channel collapse
# -----------------------------------------------------------------
def fig3_f1(report, out_prefix):
    if report is None: return
    prod = report.get("prod_167k", {})
    stage_dist = prod.get("rejection_stage_distribution", {})
    canonical_stages = [
        ("Stage_3_Classifier", "Stage 3 - RE classifier"),
        ("Stage_4_QA_Verifier", "Stage 4 - QAR verifier"),
        ("Stage_1_Gatekeeper", "Stage 1 - NER gatekeeper"),
        ("Stage_1_Verifier", "Stage 1 - NER verifier"),
        ("Stage_1_Rescue_Op", "Stage 1 - NER rescue"),
    ]
    stage_counts = [(label, stage_dist.get(key, 0)) for key, label in canonical_stages]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11.5, 4.8),
                                     gridspec_kw={"width_ratios": [1.4, 1.0]}, layout="constrained")

    ys = list(range(len(stage_counts)))
    counts = [c for _, c in stage_counts]
    ax_l.barh(ys, counts, color=PALETTE["neutral"], edgecolor="#222222", linewidth=0.8, zorder=2)
    ax_l.set_yticks(ys)
    ax_l.set_yticklabels([l for l, _ in stage_counts], fontsize=FONT_AX)
    ax_l.invert_yaxis()
    ax_l.set_xlabel("Rejection events logged at 167K", fontsize=FONT_AX, fontweight="bold")
    ax_l.set_title("(a) Input signal: stage rejections", fontsize=FONT_TITLE)

    if max(counts) > 0:
        for y, c in zip(ys, counts):
            ax_l.text(c + max(counts) * 0.02, y, f"{c:,}", va="center", fontsize=FONT_ANNOT+1)
        ax_l.set_xlim(0, max(counts) * 1.25)
    _style_axes(ax_l)

    persisted = [
        ("RE",  prod.get("re_patterns_total", 0)),
        ("QAR", prod.get("qa_patterns_total", 0)),
        ("NER", prod.get("ner_patterns_total", 0)),
    ]
    xs = list(range(len(persisted)))
    counts_r = [c for _, c in persisted]
    display_counts = [max(c, 0.7) for c in counts_r]
    colors_r = [PALETTE["anti"] if c == 0 else PALETTE["selective"] for c in counts_r]

    ax_r.bar(xs, display_counts, color=colors_r, edgecolor="#222222", linewidth=0.8, zorder=2)
    ax_r.set_yscale("log")
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels([l for l, _ in persisted], fontsize=FONT_AX)
    ax_r.set_ylabel("Persisted patterns at 167K (log scale)", fontsize=FONT_AX, fontweight="bold")
    ax_r.set_title("(b) Output state: persisted patterns", fontsize=FONT_TITLE)

    for x, c in zip(xs, counts_r):
        if c == 0:
            ax_r.text(x, 1.3, "0\n(channel\nempty)", ha="center", va="bottom",
                      fontsize=FONT_ANNOT, fontweight="bold", color=PALETTE["anti"])
        else:
            ax_r.text(x, c * 1.5, f"{c:,}", ha="center", va="bottom", fontsize=FONT_ANNOT+1)

    _style_axes(ax_r)
    ax_r.grid(True, axis="y", which="both", color=PALETTE["grid"], linewidth=0.5, linestyle=":")
    if max(counts_r) > 0: ax_r.set_ylim(0.5, max(counts_r) * 8)

    fig.suptitle("Production-scale finding: verifier-fed RE blocklist channel empty at 167K despite 785,797 Stage-3 rejections",
                 fontsize=FONT_TITLE+1)
    _save(fig, out_prefix)


# -----------------------------------------------------------------
# Figure 4: Ontology-Violation Filter
# -----------------------------------------------------------------
def fig4_ontology(out_prefix):
    triples = [
        ("tests -> associated_with -> diagnosis",  13_749),
        ("symptoms -> manifests -> diagnosis",     11_813),
        ("symptoms -> causes -> diagnosis",        11_811),
        ("treatment -> prevents -> symptoms",       6_635),
        ("history -> associated_with -> diagnosis", 4_085),
        ("diagnosis -> associated_with -> history", 1_641),
    ]
    triples.sort(key=lambda kv: -kv[1])

    fig, ax = plt.subplots(figsize=(9.0, 4.5), layout="constrained")
    ys = list(range(len(triples)))
    counts = [c for _, c in triples]

    ax.barh(ys, counts, color=PALETTE["selective"], edgecolor="#222222", linewidth=0.8, height=0.6, zorder=2)
    for y, c in zip(ys, counts):
        ax.text(c + max(counts) * 0.015, y, f"{c:,}", va="center", fontsize=FONT_ANNOT+1)

    ax.set_yticks(ys)
    ax.set_yticklabels([t for t, _ in triples], fontsize=FONT_AX, family="monospace")
    ax.invert_yaxis()
    ax.set_xlabel("Ontology-violation events on held-out cohort", fontsize=FONT_AX, fontweight="bold")
    ax.set_title("Ontology-Violation Filter: type-triples populating the RE blocklist", fontsize=FONT_TITLE)
    ax.set_xlim(0, max(counts) * 1.15)
    _style_axes(ax)

    ax.text(0.98, 0.05, f"6 patterns, {sum(counts):,} events on held-out cohort",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_ANNOT,
            color=PALETTE["neutral"], style="italic", fontweight="bold")

    _save(fig, out_prefix)


# -----------------------------------------------------------------
# Figure 5: DOWNGRADE-first measurement discipline
# -----------------------------------------------------------------
def fig5_downgrade_first(report, out_prefix):
    if report is None: return
    pair_b = report.get("qa_tagged_pass") or report.get("pair_B_v7_effect") or {}
    if not pair_b: return

    removed = [
        ("NER", abs(pair_b.get("ner_delta", 0))),
        ("RE",  abs(pair_b.get("re_delta", 0))),
        ("QAR", abs(pair_b.get("qa_delta", 0))),
    ]
    tagged = pair_b.get("v7_attributable_tags", pair_b.get("qa_tags_delta", 0))

    fig, ax = plt.subplots(figsize=(8.5, 5.0), layout="constrained")

    xs_l = [0, 1, 2]
    heights_l = [c for _, c in removed]

    # Left bars (Removed)
    bars_l = ax.bar(xs_l, heights_l, color=PALETTE["anti"], edgecolor="#222222",
                    linewidth=0.8, width=0.6, zorder=2)

    ymax = max(max(heights_l), tagged) if (heights_l or tagged) else 1
    for x, c in zip(xs_l, heights_l):
        ax.text(x, c + ymax * 0.02, f"{c:,}", ha="center", va="bottom",
                fontsize=FONT_ANNOT+1, fontweight="bold")

    # Right bar (Tagged)
    x_r = 4
    bars_r = ax.bar([x_r], [tagged], color=PALETTE["selective"], edgecolor="#222222",
                    linewidth=0.8, width=0.6, zorder=2)
    ax.text(x_r, tagged + ymax * 0.02, f"{tagged:,}", ha="center", va="bottom",
            fontsize=FONT_ANNOT+1, fontweight="bold")

    # Clean divider
    ax.axvline(3.0, color=PALETTE["grid"], linewidth=1.5, alpha=0.8, linestyle="--", zorder=1)

    x_labels = [
        "NER\n(Pairs Changed)",
        "RE\n(Pairs Changed)",
        "QAR\n(Pairs Changed)",
        "QAR\n(Gate Tagged)"
    ]
    ax.set_xticks(xs_l + [x_r])
    ax.set_xticklabels(x_labels, fontsize=FONT_AX, linespacing=1.5)

    ax.set_ylabel("Events on held-out cohort", fontsize=FONT_AX, fontweight="bold")
    ax.set_title("DOWNGRADE-first: the gate tags candidates without removing them", fontsize=FONT_TITLE)
    _style_axes(ax)

    ax.set_xlim(-0.8, x_r + 0.8)
    ax.set_ylim(0, ymax * 1.25)

    # Top headers to classify sides
    ax.text(1, ymax * 1.15, "Pairs Modified (Full-DBPM ON vs OFF)", ha="center", va="bottom",
            fontsize=FONT_AX, color=PALETTE["anti"], fontweight="bold")
    ax.text(x_r, ymax * 1.15, "Candidates Tagged", ha="center", va="bottom",
            fontsize=FONT_AX, color=PALETTE["selective"], fontweight="bold")

    fig.text(0.5, -0.02,
             "Note: The RE delta represents the deterministic Ontology-Violation Filter's suppression, not the QAR gate.",
             ha="center", fontsize=FONT_ANNOT, color=PALETTE["neutral"], style="italic")

    _save(fig, out_prefix)


# -----------------------------------------------------------------
# Main
# -----------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    rd = Path(args.reports_dir)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    print(f"Reading reports from {rd}")
    f1 = _load(rd / "F1_crosscheck.json", optional=True)
    sel5k = _load(rd / "m1v7_full_ON_selectivity.json", optional=True)
    aggr = _load(rd / "dbpm_full_aggregate.json", optional=True)

    print(f"\nRendering figures to {od}")
    fig1_iterations(od / "fig1_iteration_dissection")
    if sel5k: fig2_envelope(sel5k, od / "fig2_per_category_envelope")
    if f1: fig3_f1(f1, od / "fig3_f1_channel_collapse")
    fig4_ontology(od / "fig4_ontology_filter")
    if aggr: fig5_downgrade_first(aggr, od / "fig5_downgrade_first")

    print("\ndone")


if __name__ == "__main__":
    main()
