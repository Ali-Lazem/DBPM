# `graders/`

Scripts that reproduce the paper's reported gating numbers from the released
aggregate outputs in [`../results/`](../results/). None requires a GPU or the source
corpus; each reads a JSON of counts and rates and prints the corresponding paper
figure.

| File | Reproduces | Reads |
| --- | --- | --- |
| `grade_m1v7_selectivity.py` | The five-signal selectivity dissection: per-signal lift, and the NER-coverage path lift (1.84) with per-category breakdown (Table 6, Figure 4) | `results/m1v7_full_ON_selectivity.json` |
| `compute_v7_lift_ci.py` | The path-lift 95% confidence interval [1.75, 1.93] and the test against lift = 1 (§6.3) | `results/m1v7_full_ON_selectivity.json` |
| `crosscheck_F1_167k_blocklist.py` | The production-scale finding: the empty relation channel against 785,797 logged rejections (§6.1, Figure 3) | `results/F1_crosscheck.json` |
| `bootstrap_v7_lift_ci.py` | The bootstrap confidence interval on the path lift, as a resampling cross-check of the analytic CI | `results/m1v7_full_ON_selectivity.json` |
| `aggregate_dbpm_provenance.py` | The aggregate DBPM provenance: gating-event counts and the 2.44% of pipeline pairs touched (§6.5) | `results/dbpm_full_aggregate.json` |

## Usage

```bash
# every grader takes the released JSON it reads and prints the paper number:
python3 graders/grade_m1v7_selectivity.py --json results/m1v7_full_ON_selectivity.json
python3 graders/compute_v7_lift_ci.py    --json results/m1v7_full_ON_selectivity.json
python3 graders/crosscheck_F1_167k_blocklist.py
```

## Note

These scripts grade **released aggregate outputs**, not raw clinical text. Every
number they print traces to a JSON in [`../results/`](../results/); no patient data is
read or required. The cohorts these outputs were computed over are indexed by the UID
lists in [`../cohorts/`](../cohorts/).

<!-- CONFIRM: list matches the actual files committed in graders/. Add or remove rows
     to match. If a grader takes different flags than --json, correct the Usage block. -->
