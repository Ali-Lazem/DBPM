#!/usr/bin/env python3
"""
compute_v7_lift_ci.py
=====================

Reads the existing m1v7_locked_5k.json (or equivalent) report produced
by grade_m1v7_selectivity.py, restricts to the nine v7-path categories,
and computes:

  - Point estimate of the v7-path lift
  - Delta-method (analytic) 95% CI on the lift, using the standard
    risk-ratio variance formula:
        Var[log(RR)] = (1 - p1)/(n1 * p1) + (1 - p0)/(n0 * p0)
  - Test of H0: lift = 1 (no selectivity)

This is the citation-defensible CI for the paper's selectivity claim.
A bootstrap is not required and would be less defensible here because
FAIL pairs in this pipeline are stored as aggregated bpm.tasks.qa.patterns
without per-patient breakdown; the delta-method approach uses only the
sufficient statistics that grade_m1v7_selectivity.py already produces.

Usage
-----
    python3 compute_v7_lift_ci.py --report path/to/reports/m1v7_locked_5k.json
"""
import argparse, json, math, sys
from pathlib import Path

# The 9 QAR categories on which the v7 path can fire
# (adverse_event and outcome_mortality fall through to v6 fallback;
#  excluded from the v7-path lift calculation)
V7_PATH_CATS = {
    "diagnosis", "treatment", "symptoms", "tests", "history",
    "observation", "risk_factor", "outcome_clinicalstatus",
    "outcome_disposition",
}


def normal_two_sided_p(z):
    """Two-sided p-value from a z statistic (no scipy required)."""
    return 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--report",
        required=True,
        help="Path to m1v7_locked_5k.json (output of grade_m1v7_selectivity.py)",
    )
    args = ap.parse_args()

    p = Path(args.report)
    if not p.exists():
        print(f"[FATAL] report not found: {p}")
        sys.exit(2)

    report = json.loads(p.read_text())
    per_cat = report.get("per_category", [])

    # Restrict to v7-path categories
    v7_rows = [r for r in per_cat if r["category"] in V7_PATH_CATS]
    missing = V7_PATH_CATS - {r["category"] for r in v7_rows}
    if missing:
        print(f"[WARN] v7-path categories missing from report: {sorted(missing)}")

    # Aggregate sufficient statistics
    n_pass = sum(r["n_pass"] for r in v7_rows)
    n_fail = sum(r["n_fail"] for r in v7_rows)
    pg = sum(r["tagged_pass"] for r in v7_rows)
    fg = sum(r["tagged_fail"] for r in v7_rows)

    if n_pass == 0 or n_fail == 0 or pg == 0 or fg == 0:
        print(f"[FATAL] one of the counts is zero: "
              f"n_pass={n_pass}, n_fail={n_fail}, pg={pg}, fg={fg}")
        sys.exit(3)

    p_tag_pass = pg / n_pass
    p_tag_fail = fg / n_fail
    lift = p_tag_fail / p_tag_pass

    # Delta-method variance of log(risk ratio)
    var_log_lift = (1 - p_tag_fail) / (n_fail * p_tag_fail) \
                 + (1 - p_tag_pass) / (n_pass * p_tag_pass)
    se_log_lift = math.sqrt(var_log_lift)

    z = 1.96
    log_lift = math.log(lift)
    ci_lo = math.exp(log_lift - z * se_log_lift)
    ci_hi = math.exp(log_lift + z * se_log_lift)

    # Test of H0: lift = 1 (no selectivity)
    z_stat = log_lift / se_log_lift
    p_value = normal_two_sided_p(z_stat)

    print("=" * 72)
    print("v7-path lift: delta-method 95% CI")
    print("=" * 72)
    print(f"  Categories in v7-path  : {sorted(r['category'] for r in v7_rows)}")
    print(f"  n_PASS                 : {n_pass:,}")
    print(f"  n_FAIL                 : {n_fail:,}")
    print(f"  PASS pairs tagged      : {pg:,}  (P(tag|PASS) = {p_tag_pass:.4f})")
    print(f"  FAIL pairs tagged      : {fg:,}  (P(tag|FAIL) = {p_tag_fail:.4f})")
    print()
    print(f"  Lift (point estimate)  : {lift:.4f}")
    print(f"  SE[log(lift)]          : {se_log_lift:.5f}")
    print(f"  95% CI (delta method)  : [{ci_lo:.3f}, {ci_hi:.3f}]")
    print()
    print(f"  Test of H0: lift = 1   : z = {z_stat:.2f}, p = {p_value:.2e}")
    print()
    print("For the paper:")
    print(f"  v7-path lift = {lift:.2f}, 95% CI [{ci_lo:.2f}, {ci_hi:.2f}]")


if __name__ == "__main__":
    main()
