"""
Bootstrap CI for v7-path lift at sealed 5K.
Patient-level resampling (1000 iterations) to respect cohort structure.
Uses the gate-decision graded output for the v7-only ablation arm.
"""
import json, random
import numpy as np
from pathlib import Path

# Path to the per-pair output from the v7-only ON arm
ENRICHED = Path("path/to/code/OUTPUT_DIR_ON/multitask_data_enriched.jsonl")
V7_PATH_CATS = {"diagnosis", "treatment", "symptoms", "tests", "history",
                "observation", "risk_factor", "outcome_clinicalstatus",
                "outcome_disposition"}  # 9 v7-mapped categories

# Load per-pair records: each has patient_uid, category, verifier_verdict, dbpm_gate
pairs = []
with open(ENRICHED) as f:
    for line in f:
        rec = json.loads(line)
        for qa in rec.get("qa_pairs", []):
            cat = qa.get("category")
            if cat not in V7_PATH_CATS:
                continue
            pairs.append({
                "uid": rec["patient_uid"],
                "cat": cat,
                "fail": qa.get("verifier_verdict") == "FAIL",
                "tagged": qa.get("dbpm_gate") in {"DOWNGRADE", "BLOCK"},
            })

# Group by patient for patient-level bootstrap
by_uid = {}
for p in pairs:
    by_uid.setdefault(p["uid"], []).append(p)
uids = list(by_uid.keys())
print(f"v7-path pairs: {len(pairs)}, patients: {len(uids)}")

def lift(pair_list):
    fail = [p for p in pair_list if p["fail"]]
    pass_ = [p for p in pair_list if not p["fail"]]
    if not fail or not pass_:
        return None
    p_tag_fail = sum(p["tagged"] for p in fail) / len(fail)
    p_tag_pass = sum(p["tagged"] for p in pass_) / len(pass_)
    if p_tag_pass == 0:
        return None
    return p_tag_fail / p_tag_pass

# Point estimate
point = lift(pairs)
print(f"Point estimate: lift = {point:.4f}")

# Bootstrap (patient-level resampling, 1000 iters)
rng = random.Random(42)
boot_lifts = []
for i in range(1000):
    sampled_uids = [rng.choice(uids) for _ in range(len(uids))]
    sampled_pairs = []
    for u in sampled_uids:
        sampled_pairs.extend(by_uid[u])
    L = lift(sampled_pairs)
    if L is not None:
        boot_lifts.append(L)

ci_lo, ci_hi = np.percentile(boot_lifts, [2.5, 97.5])
print(f"95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]  (n_bootstrap={len(boot_lifts)})")
