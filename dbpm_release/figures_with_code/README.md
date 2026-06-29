# `figures_with_code/`

The scripts that generate the paper's figures, with the rendered outputs. Every
figure in the paper is produced here, so a reader can regenerate all of them from the
released aggregate data without the extraction pipeline.

The manuscript's `\includegraphics` calls expect `fig_1.pdf` … `fig_5.pdf`; the files
here use that scheme.

## Figures and the code that produces them

| Figure (paper) | What it shows | Output | Needs |
| --- | --- | --- | --- |
| Fig. 4 | Five candidate signals for the QA gate; four fail, one is selective (lift 1.84) | `fig_1.pdf` | nothing (values inline) |
| Fig. 5 | Per-category operating envelope of the NER-coverage gate (log-scale lift, coverage) | `fig_2.pdf` | nothing (values inline) |
| Fig. 3 | Verifier-fed channel collapse at 167K: 785,797 rejections vs 0 persisted RE patterns | `fig_3.pdf` | `results/F1_crosscheck.json` |
| Fig. (ontology) | Type-triples populating the RE blocklist via the Ontology-Violation Filter | `fig_4.pdf` | nothing (values inline) |
| Fig. (DOWNGRADE-first) | Pair-count deltas vs gate-tag counts (tags without removing) | `fig_5.pdf` | `results/dbpm_full_aggregate.json` |

All five are rendered by one script: **`make_paper_figures.py`**.

## Reproducing the figures

```bash
pip install matplotlib numpy

# render every figure (fig_1 … fig_5) in one run:
python3 make_paper_figures.py --reports-dir ../results --out-dir .
```

Each figure is written as both a `.pdf` (for the manuscript) and a `.png` (for quick
viewing). `fig_1`, `fig_2`, and `fig_4` carry their values inline (the figures whose
numbers are fixed in the paper) and render with no input file. `fig_3` and `fig_5`
read their aggregate JSON from `--reports-dir`; if a file is absent the script prints
a `[skip]` line and continues, so the inline figures always render.

## Notes

- A single fixed publication style (palette, fonts, sizes) is defined at the top of
  the script, so every figure matches the manuscript.
- No GPU and no clinical text are required: the data-driven figures are built from the
  aggregate counts in [`../results/`](../results/) (counts and rates only).
- In-figure vocabulary matches the paper: "candidate signals," "NER-coverage gate,"
  "held-out cohort."

<!-- CONFIRM:
  (1) make_paper_figures.py is the single committed master script (replacing any older
      make_paper_figures.py / make_fig_1.py / make_fig_2.py versions).
  (2) The committed outputs are fig_1.pdf … fig_5.pdf (delete any old descriptive-named
      files: fig1_iteration_dissection.*, fig2_per_category_envelope.*,
      fig3_f1_channel_collapse.*, fig4_ontology_filter.*, fig5_downgrade_first.*).
  (3) The two results JSONs (F1_crosscheck.json, dbpm_full_aggregate.json,
      m1v7_full_ON_selectivity.json) exist in ../results/ with these exact names.
-->
