# `cohorts/`

Patient UID lists that index the **public PMC-Patients corpus** (Zhao et al., 2023).
These files contain identifiers only, no clinical text, and they fix the exact patient
sets used for the paper's ablations so the released aggregate results in
[`../results/`](../results/) are reproducible against a known cohort.

| File | Cohort | Used for |
| --- | --- | --- |
| `ablation_5k_uids.json` | The held-out 5,000-patient cohort: stratified random subsample, fixed across all ablation arms | The paired ablations of §5.2 and the selectivity measurements of §6.3–6.5 |
| `dev_cohort_uids.json` | The development cohort | The read-only hypothesis test that motivated the NER-coverage gate, and the development-cohort replication (§6.3) |

## Stratification

The held-out cohort is stratified along three dimensions, age bucket, gender, and
document-length tertile, yielding 24 non-empty strata, with a floor allocation per
stratum so rare strata remain visible (paper §5.1). The file records the per-stratum
allocation and the final shuffled UID order alongside the identifiers.

## Using these lists

The UIDs map directly onto PMC-Patients records. To reconstruct a cohort, load the
corresponding PMC-Patients narratives by UID; the corpus is publicly available from
its original release. No identifiers in these files reveal patient information beyond
the public corpus's own record keys.

<!-- CONFIRM: filenames match what is committed in cohorts/. Paper refers to the
     held-out list as ablation_5k_uids.json; adjust if the committed name differs. -->
