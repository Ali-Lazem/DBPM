# `results/`

Released aggregate gating-outcome outputs. **Counts and rates only, no clinical
text.** Every quantitative claim in the paper traces to a file here, and the scripts
in [`../graders/`](../graders/) reproduce the reported numbers from these files.

| File | Contents | Used by |
| --- | --- | --- |
| `m1v7_full_ON_selectivity.json` | Per-signal and per-category selectivity for the NER-coverage gate: tag counts on PASS and FAIL pairs, the path lift, and the per-category rows behind Table 6 | `grade_m1v7_selectivity.py`, `compute_v7_lift_ci.py`, `bootstrap_v7_lift_ci.py` |
| `F1_crosscheck.json` | Production-scale (167,034-patient) channel state: rejection-stream composition by stage and the persisted-pattern counts per channel (relation 0, NER 5,860, QA 11) | `crosscheck_F1_167k_blocklist.py` |
| `dbpm_full_aggregate.json` | Aggregate DBPM provenance from the master-switch ablation: per-stage pair-count deltas, attributable QA tags, and the combined gating-event total | `aggregate_dbpm_provenance.py` |

## What these files are

Each JSON is an **aggregate summary** computed from the deployed run's outputs: it
records how many candidates were tagged, blocked, or allowed, broken down by task,
category, and verifier verdict. It contains no questions, answers, entities, or
relations, only the counts and rates needed to reproduce the paper's quantitative
results. The patient cohorts these aggregates summarise are identified by the UID
lists in [`../cohorts/`](../cohorts/), which index the public PMC-Patients corpus; the
derived extraction corpus itself is not released.

<!-- CONFIRM: filenames match what is actually committed in results/. The three above
     are the outputs the main-README reproduce-table and the graders reference. -->
