#!/usr/bin/env python3
"""
grade_m1v7_selectivity.py
==========================

Extends grade_m1v6 with NER-coverage stratification. The v7 patch
writes m1v7_has_ner_coverage on each qa_entry; this grader reads it
back and computes:

  - Global lift = P(tag|FAIL) / P(tag|PASS)
  - Per-category lift
  - NER-coverage validation: among PASS pairs, does the rate of
    has_ner_coverage match the rate among FAIL? If has_ner_coverage
    is genuinely predictive of PASS (the v7 hypothesis), PASS pairs
    should have substantially higher coverage rate than FAILs.

Reads enriched (PASS) and bpm.qa.patterns (FAIL aggregated). FAIL
side does not have per-pair has_ner_coverage; only the category-level
aggregate. So has_ner_coverage validation is done on the PASS side
and corroborated via the gate-tagging asymmetry.

Usage
-----
    python grade_m1v7_selectivity.py \\
        --enr <enriched.jsonl> \\
        --bpm <bpm_production.json> \\
        --out <report.json>
"""
import argparse, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path


QA_NO_NER_BUCKET = {"adverse_event", "outcome_mortality"}


def _tagged(g):
    return str(g or "none").lower() in ("block", "downgrade")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enr", required=True)
    ap.add_argument("--bpm", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    enr_p = Path(args.enr); bpm_p = Path(args.bpm)
    for p in (enr_p, bpm_p):
        if not p.exists():
            print(f"[FATAL] missing: {p}"); sys.exit(2)

    # PASS side
    pt = pg = 0
    by_cat = defaultdict(lambda: {"pt":0, "pg":0, "ft":0, "fg":0})
    has_coverage_dist = Counter()  # global: pass-side coverage flag
    cov_by_cat = defaultdict(lambda: {"covered_pass":0, "uncovered_pass":0})
    n_with_v7_field = 0
    with open(enr_p) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: rec = json.loads(line)
            except: continue
            for q in (rec.get("tasks",{}) or {}).get("qa",[]) or []:
                if not isinstance(q,dict): continue
                cat = str(q.get("category","") or "").lower().strip()
                if not cat: continue
                gate = q.get("dbpm_gate","none")
                if "PASS" not in str(q.get("verifier_status","")).upper():
                    continue
                pt += 1
                by_cat[cat]["pt"] += 1
                if _tagged(gate):
                    pg += 1
                    by_cat[cat]["pg"] += 1
                if "m1v7_has_ner_coverage" in q:
                    n_with_v7_field += 1
                    covered = bool(q["m1v7_has_ner_coverage"])
                    has_coverage_dist[covered] += 1
                    if covered:
                        cov_by_cat[cat]["covered_pass"] += 1
                    else:
                        cov_by_cat[cat]["uncovered_pass"] += 1

    # FAIL side
    bpm = json.loads(bpm_p.read_text())
    ft = fg = 0
    for p in (bpm.get("tasks",{}) or {}).get("qa",{}).get("patterns",[]) or []:
        if not isinstance(p,dict): continue
        if not p.get("error_class"): continue
        cat = str(p.get("category","") or "").lower().strip()
        if not cat: continue
        n = int(p.get("count",0) or 0)
        if n <= 0: continue
        gate = p.get("dbpm_gate","none")
        ft += n
        by_cat[cat]["ft"] += n
        if _tagged(gate):
            fg += n
            by_cat[cat]["fg"] += n

    p_tag_pass = pg/pt if pt else None
    p_tag_fail = fg/ft if ft else None
    if p_tag_fail is not None and p_tag_pass is not None and p_tag_pass > 0:
        lift = p_tag_fail / p_tag_pass
    elif p_tag_pass == 0 and p_tag_fail and p_tag_fail > 0:
        lift = float("inf")
    else:
        lift = None

    # Per-category
    per_cat = []
    for cat in sorted(by_cat.keys()):
        d = by_cat[cat]
        if d["pt"] < 5 or d["ft"] < 5: continue
        pp = d["pg"]/d["pt"]; pf = d["fg"]/d["ft"]
        if pp > 0:
            l = pf/pp
        elif pf > 0:
            l = float("inf")
        else:
            l = None
        fr = d["ft"]/(d["ft"]+d["pt"]) if (d["ft"]+d["pt"]) else 0
        cov = cov_by_cat.get(cat, {"covered_pass":0, "uncovered_pass":0})
        per_cat.append({
            "category": cat, "n_pass": d["pt"], "n_fail": d["ft"],
            "fail_rate": round(fr, 4),
            "tagged_pass": d["pg"], "tagged_fail": d["fg"],
            "p_tag_given_pass": round(pp, 4),
            "p_tag_given_fail": round(pf, 4),
            "lift": (round(l,4) if isinstance(l,(int,float))
                     and not math.isinf(l) else
                     ("inf" if l == float("inf") else None)),
            "pass_with_ner_coverage": cov["covered_pass"],
            "pass_without_ner_coverage": cov["uncovered_pass"],
            "no_ner_bucket": cat in QA_NO_NER_BUCKET,
        })
    per_cat.sort(key=lambda r: -(r["fail_rate"] or 0))

    report = {
        "_provenance": {
            "enriched": str(enr_p), "bpm": str(bpm_p),
            "n_pass_with_v7_field": n_with_v7_field,
            "n_pass_total": pt, "n_fail_total": ft,
        },
        "global": {
            "pass_total": pt, "pass_tagged": pg,
            "fail_total": ft, "fail_tagged": fg,
            "p_tag_given_pass": (round(p_tag_pass,4) if p_tag_pass else None),
            "p_tag_given_fail": (round(p_tag_fail,4) if p_tag_fail else None),
            "lift": (round(lift,4) if isinstance(lift,(int,float))
                     and not math.isinf(lift) else
                     ("inf" if lift == float("inf") else None)),
            "pass_coverage_distribution": {
                "covered": has_coverage_dist[True],
                "uncovered": has_coverage_dist[False],
            },
        },
        "per_category": per_cat,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))

    print("=" * 78)
    print("M1-v7 selectivity (neuro-symbolic intersection)")
    print("=" * 78)
    print(f"\n  PASS pairs in enriched : {pt}")
    print(f"  PASS pairs with v7 field: {n_with_v7_field}")
    print(f"  FAIL records in bpm     : {ft}")
    if n_with_v7_field == 0:
        print("\n  ⚠ NO m1v7_has_ner_coverage field on any pair.")
        print("    The v7 patch did not run (or fell through the except).")
        sys.exit(3)
    print(f"\n  PASS coverage: covered={has_coverage_dist[True]}, "
          f"uncovered={has_coverage_dist[False]}")
    print(f"  PASS tagged   : {pg} ({100*pg/pt:.2f}%)" if pt else "")
    print(f"  FAIL tagged   : {fg} ({100*fg/ft:.2f}%)" if ft else "")
    if p_tag_pass is not None: print(f"\n  P(tag|PASS) = {p_tag_pass:.4f}")
    if p_tag_fail is not None: print(f"  P(tag|FAIL) = {p_tag_fail:.4f}")
    if isinstance(lift,(int,float)) and not math.isinf(lift):
        print(f"  LIFT        = {lift:.4f}")
    elif lift == float("inf"):
        print(f"  LIFT        = inf")
    else:
        print(f"  LIFT        = undefined")

    print("\n  Per-category:")
    if per_cat:
        print(f"  {'category':<24} {'pass':>5} {'fail':>5} "
              f"{'rate':>5} {'P(t|P)':>7} {'P(t|F)':>7} {'lift':>7} {'cov%':>5}")
        print("  " + "-"*78)
        for r in per_cat:
            ls = (f"{r['lift']:>7.3f}" if isinstance(r['lift'],(int,float))
                  else (f"{'(undef)':>7}" if r['lift'] is None
                        else f"{r['lift']:>7}"))
            cov_pct = (100 * r['pass_with_ner_coverage'] /
                       max(r['pass_with_ner_coverage'] +
                           r['pass_without_ner_coverage'], 1))
            note = " (no_NER)" if r["no_ner_bucket"] else ""
            print(f"  {r['category']:<24} {r['n_pass']:>5d} "
                  f"{r['n_fail']:>5d} {r['fail_rate']:>5.2f} "
                  f"{r['p_tag_given_pass']:>7.4f} "
                  f"{r['p_tag_given_fail']:>7.4f} {ls} "
                  f"{cov_pct:>5.1f}{note}")

    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    if lift is None:
        print("  undefined")
    elif lift == float("inf"):
        print("  Lift = inf — maximum selectivity")
    elif lift >= 1.5:
        print(f"  ✓ Lift = {lift:.3f} >= 1.5 — v7 IS SELECTIVE.")
        print(f"    Neuro-symbolic intersection signal recovered selectivity")
        print(f"    that v3-v6 could not. The hypothesis is empirically")
        print(f"    confirmed at dev-mini scale.")
    elif lift >= 1.1:
        print(f"  ◐ Lift = {lift:.3f} — weakly selective.")
        print(f"    Modest signal; consider promotion to N=2000 for")
        print(f"    power-adequate measurement.")
    elif lift >= 0.9:
        print(f"  ✗ Lift = {lift:.3f} ≈ 1 — no predictive power.")
        print(f"    The dev-mini-validated hypothesis did not translate")
        print(f"    to instance-level selectivity.")
    else:
        print(f"  ✗✗ Lift = {lift:.3f} < 1 — anti-selective.")
        print(f"    Architecture has structural limit. Ship Path A.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
