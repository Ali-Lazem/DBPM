#!/usr/bin/env python3
"""
make_paper_figures.py  -- master figure script for the DBPM paper
=================================================================
Renders all five paper figures in one run, with output names matching the
manuscript includegraphics calls (fig_1.pdf ... fig_5.pdf).

Figure map (paper -> function -> output):
  Fig. 4  five candidate signals for the QA gate      fig1_signals()    -> fig_1
  Fig. 5  per-category envelope (NER-coverage gate)    fig2_envelope()   -> fig_2
  Fig. 3  verifier-fed channel collapse at 167K        fig3_channel()    -> fig_3
  Fig.    ontology type-triples populating RE blocklist fig4_ontology()  -> fig_4
  Fig.    DOWNGRADE-first: deltas vs gate-tag counts    fig5_downgrade()  -> fig_5

fig_1 and fig_2 are the redesigned versions (horizontal bars; log-scale envelope;
vocabulary aligned to the paper: "candidate signals", "NER-coverage gate",
"held-out cohort"). fig_3/4/5 are unchanged in design; only their output names
follow the fig_N scheme.

Usage:
  python3 make_paper_figures.py --reports-dir ../results --out-dir .
  # fig_1 and fig_4 carry locked values inline and render without any input file;
  # fig_2, fig_3, fig_5 read their JSON from --reports-dir (skipped if absent).
"""
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


# -- Professional publication style ----------------------------------
PALETTE = {
    "selective":   "#0072B2",  # deep scientific blue
    "marginal":    "#E69F00",  # muted gold/orange
    "anti":        "#D55E00",  # vermillion / brick red
    "absent":      "#999999",  # neutral slate gray
    "categorical": "#CC79A7",  # soft magenta/purple
    "saturated":   "#56B4E9",  # sky blue
    "fallback":    "#999999",  # gray (no-NER-bucket)
    "no_bucket":   "#E5E5E5",  # light background gray
    "neutral":     "#4A4A4A",  # charcoal
    "axis":        "#2B2B2B",  # off-black
    "grid":        "#D1D1D1",  # soft grid
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


def _load(p, optional=False):
    if not p.exists():
        if optional:
            print(f"  [skip] missing report: {p.name}")
            return None
        print(f"[FATAL] missing report: {p}")
        sys.exit(2)
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError as e:
        print(f"[FATAL] JSON parse {p}: {e}")
        sys.exit(2)


# =================================================================
# Fig. 4 in the paper -> fig_1 : five candidate signals (REDESIGNED)
# =================================================================
def fig1_signals(out_prefix):
    # (name, lift, failure_mode_label, is_selective)
    signals = [
        ("Category-aggregate fail-mass", 0.006, "aggregation-level mismatch", False),
        ("Verifier uncertainty",          None, "signal absence (flag unpopulated)", False),
        ("Per-category fail-rate",        0.05, "counting asymmetry", False),
        ("Per-patient difficulty",        0.00, "cross-task orthogonality", False),
        ("NER-coverage intersection",     1.84, "structurally aligned (selective)", True),
    ]

    fig, ax = plt.subplots(figsize=(9.2, 4.8), layout="constrained")
    ys = list(range(len(signals)))[::-1]
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

    ax.axvline(threshold, color=PALETTE["selective"], linestyle="--",
               linewidth=1.3, alpha=0.85, zorder=2)
    ax.axvspan(threshold, 2.15, color=PALETTE["selective"], alpha=0.05, zorder=0)
    ax.text(threshold + 0.03, 1.15, "selectivity\nthreshold (1.5)",
            fontsize=FONT_ANNOT, color=PALETTE["selective"],
            fontweight="bold", va="center", ha="left")

    ax.set_yticks(ys)
    ax.set_yticklabels([s[0] for s in signals], fontsize=FONT_AX)
    ax.set_ylim(-0.6, len(signals) - 0.4)
    ax.set_xlim(0, 2.15)
    ax.set_xlabel("Selectivity lift  =  P(tag | FAIL) / P(tag | PASS)",
                  fontsize=FONT_AX, fontweight="bold")
    ax.set_title("Five candidate signals for the question-answering gate",
                 fontsize=FONT_TITLE, pad=10)
    _style_axes(ax)
    ax.grid(False)
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.6,
            linestyle=":", zorder=0)
    _save(fig, out_prefix)


# =================================================================
# Fig. 5 in the paper -> fig_2 : per-category envelope (REDESIGNED)
# =================================================================
def fig2_envelope(out_prefix, report=None):
    # Locked per-category values from Table 6 (used if no report supplied).
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

    FLOOR = 0.04
    fig, ax = plt.subplots(figsize=(9.4, 5.4), layout="constrained")
    ys = list(range(len(rows)))[::-1]

    for y, (cat, lift, cov, regime) in zip(ys, rows):
        color = PALETTE[regime]
        if lift is None or lift == 0.0:
            xval = FLOOR
            bar_alpha = 0.35
        else:
            xval = lift
            bar_alpha = 1.0
        ax.barh(y, xval, height=0.6, color=color, edgecolor="#222222",
                linewidth=0.7, alpha=bar_alpha, zorder=3)

        if lift is None:
            vtxt = "no tagging"
        elif lift == 0.0:
            vtxt = "fallback (lift 0.00)"
        else:
            vtxt = f"lift {lift:.2f}"
        ax.text(xval * 1.12 if xval > FLOOR else FLOOR * 1.3, y,
                f"{vtxt}   cov {cov:.0f}%", va="center", ha="left",
                fontsize=FONT_ANNOT, color=PALETTE["axis"])

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


# =================================================================
# Fig. 3 in the paper -> fig_3 : verifier-fed channel collapse
# =================================================================
def fig3_channel(report, out_prefix):
    if report is None:
        print("  [skip] fig_3 needs F1_crosscheck.json")
        return
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
    ax_l.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.6, linestyle=":", zorder=0)

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


# =================================================================
# Fig. ontology -> fig_4 : type-triples populating the RE blocklist
# =================================================================
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
    ax.grid(True, axis="x", color=PALETTE["grid"], linewidth=0.6, linestyle=":", zorder=0)
    ax.text(0.98, 0.05, f"6 patterns, {sum(counts):,} events on held-out cohort",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT_ANNOT,
            color=PALETTE["neutral"], style="italic", fontweight="bold")
    _save(fig, out_prefix)


# =================================================================
# Fig. downgrade -> fig_5 : pair-count deltas vs gate-tag counts
# =================================================================
def fig5_downgrade(report, out_prefix):
    if report is None:
        print("  [skip] fig_5 needs dbpm_full_aggregate.json")
        return
    pair_b = report.get("qa_tagged_pass") or report.get("pair_B_v7_effect") or {}
    if not pair_b:
        print("  [skip] fig_5: aggregate JSON has no qa_tagged_pass/pair_B_v7_effect block")
        return

    removed = [
        ("NER", abs(pair_b.get("ner_delta", 0))),
        ("RE",  abs(pair_b.get("re_delta", 0))),
        ("QAR", abs(pair_b.get("qa_delta", 0))),
    ]
    tagged = pair_b.get("v7_attributable_tags", pair_b.get("qa_tags_delta", 0))

    fig, ax = plt.subplots(figsize=(8.5, 5.0), layout="constrained")
    xs_l = [0, 1, 2]
    heights_l = [c for _, c in removed]
    ax.bar(xs_l, heights_l, color=PALETTE["anti"], edgecolor="#222222",
           linewidth=0.8, width=0.6, zorder=2)
    ymax = max(max(heights_l), tagged) if (heights_l or tagged) else 1
    for x, c in zip(xs_l, heights_l):
        ax.text(x, c + ymax * 0.02, f"{c:,}", ha="center", va="bottom",
                fontsize=FONT_ANNOT+1, fontweight="bold")
    x_r = 4
    ax.bar([x_r], [tagged], color=PALETTE["selective"], edgecolor="#222222",
           linewidth=0.8, width=0.6, zorder=2)
    ax.text(x_r, tagged + ymax * 0.02, f"{tagged:,}", ha="center", va="bottom",
            fontsize=FONT_ANNOT+1, fontweight="bold")
    ax.axvline(3.0, color=PALETTE["grid"], linewidth=1.5, alpha=0.8, linestyle="--", zorder=1)

    x_labels = ["NER\n(Pairs Changed)", "RE\n(Pairs Changed)",
                "QAR\n(Pairs Changed)", "QAR\n(Gate Tagged)"]
    ax.set_xticks(xs_l + [x_r])
    ax.set_xticklabels(x_labels, fontsize=FONT_AX, linespacing=1.5)
    ax.set_ylabel("Events on held-out cohort", fontsize=FONT_AX, fontweight="bold")
    ax.set_title("DOWNGRADE-first: the gate tags candidates without removing them", fontsize=FONT_TITLE)
    _style_axes(ax)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.6, linestyle=":", zorder=0)
    ax.set_xlim(-0.8, x_r + 0.8)
    ax.set_ylim(0, ymax * 1.25)
    ax.text(1, ymax * 1.15, "Pairs Modified (Full-DBPM ON vs OFF)", ha="center", va="bottom",
            fontsize=FONT_AX, color=PALETTE["anti"], fontweight="bold")
    ax.text(x_r, ymax * 1.15, "Candidates Tagged", ha="center", va="bottom",
            fontsize=FONT_AX, color=PALETTE["selective"], fontweight="bold")
    fig.text(0.5, -0.02,
             "Note: The RE delta represents the deterministic Ontology-Violation Filter's suppression, not the QAR gate.",
             ha="center", fontsize=FONT_ANNOT, color=PALETTE["neutral"], style="italic")
    _save(fig, out_prefix)


# =================================================================
# Main
# =================================================================
def main():
    ap = argparse.ArgumentParser(description="Render all DBPM paper figures (fig_1..fig_5).")
    ap.add_argument("--reports-dir", default="../results",
                    help="folder with the aggregate JSONs (default: ../results)")
    ap.add_argument("--out-dir", default=".",
                    help="where to write fig_1..fig_5 (default: current dir)")
    args = ap.parse_args()

    rd = Path(args.reports_dir)
    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)

    print(f"Reading reports from {rd}")
    f1    = _load(rd / "F1_crosscheck.json", optional=True)
    sel5k = _load(rd / "m1v7_full_ON_selectivity.json", optional=True)
    aggr  = _load(rd / "dbpm_full_aggregate.json", optional=True)

    print(f"\nRendering figures to {od}")
    fig1_signals(od / "fig_1")                  # paper Fig. 4  (locked values, inline)
    fig2_envelope(od / "fig_2", report=sel5k)   # paper Fig. 5  (locked values, inline)
    fig3_channel(f1, od / "fig_3")              # paper Fig. 3  (needs F1_crosscheck.json)
    fig4_ontology(od / "fig_4")                 # paper ontology fig (locked values, inline)
    fig5_downgrade(aggr, od / "fig_5")          # paper downgrade fig (needs aggregate JSON)

    print("\ndone")


if __name__ == "__main__":
    main()
