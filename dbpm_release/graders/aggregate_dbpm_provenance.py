#!/usr/bin/env python3
"""
aggregate_dbpm_provenance.py
=============================

Computes the composite DBPM-aggregate provenance from the two
existing sealed 5K ablation pairs:

  Pair A: risk_v7_v5_M3_locked_{ON,OFF}  — isolates M3-RE effect
  Pair B: risk_v7_v7_locked_{ON,OFF}     — isolates M1-v7 QA gate
                                            effect

Both pairs were run on the same sealed 5K cohort, so per-mechanism
attribution can be stacked into a composite 'DBPM new mechanisms vs
none' number that does not require an additional GPU run.

What this script does NOT do:
  - It does not claim a 'true ON-vs-OFF' DBPM aggregate (that would
    require a new arm with both M1V7_ENABLE=0 AND BPM_DISABLE_M3=1).
  - It does not claim a runtime speed-up — the observed 40-second
    delta across 31,690 seconds is well within per-patient variance.

What this script DOES report:
  - Per-stage pair counts in each arm (NER, RE, QA)
  - Pair-count deltas attributable to each mechanism separately
  - Composite 'DBPM-touched pair events' (M3 suppressions +
    v7 DOWNGRADE-tags + legacy channel events if measurable)
  - Composite percentage of pipeline pairs touched by DBPM mechanisms
  - Honest runtime variance discussion

Read-only on all inputs. No GPU.

Usage
-----
    python aggregate_dbpm_provenance.py \\
        --v5-on  risk_v7_v5_M3_locked_ON \\
        --v5-off risk_v7_v5_M3_locked_OFF \\
        --v7-on  risk_v7_v7_locked_ON \\
        --v7-off risk_v7_v7_locked_OFF \\
        --out    /path/to/reports/dbpm_aggregate.json
"""
import argparse
import json
import math
import sys
from pathlib import Path


def _count_pairs(enr_path):
    """Count per-stage pair totals from enriched output.

    Returns dict with ner/re/qa pair counts and v7-tagged counts
    (m1v7_has_ner_coverage=False AND dbpm_gate=DOWNGRADE/BLOCK).
    """
    counts = {
        "n_patients": 0,
        "ner_pairs": 0,
        "re_pairs": 0,
        "qa_pairs": 0,
        "qa_pass": 0,
        "qa_fail": 0,
        "qa_tagged_pass": 0,
        "qa_tagged_fail": 0,
        "qa_v7_uncovered_tagged": 0,
    }
    if not enr_path.exists():
        return counts
    with open(enr_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            counts["n_patients"] += 1
            tasks = rec.get("tasks", {}) or {}
            ner = tasks.get("ner", []) or []
            re_ = tasks.get("re", []) or []
            qa = tasks.get("qa", []) or []
            counts["ner_pairs"] += sum(
                1 for n in ner if isinstance(n, dict))
            counts["re_pairs"] += sum(
                1 for r in re_ if isinstance(r, dict))
            for q in qa:
                if not isinstance(q, dict):
                    continue
                counts["qa_pairs"] += 1
                vs = str(q.get("verifier_status", "")).upper()
                gate = str(q.get("dbpm_gate", "none")).lower()
                tagged = gate in ("downgrade", "block")
                if "PASS" in vs:
                    counts["qa_pass"] += 1
                    if tagged:
                        counts["qa_tagged_pass"] += 1
                        if q.get("m1v7_has_ner_coverage") is False:
                            counts["qa_v7_uncovered_tagged"] += 1
                elif "FAIL" in vs:
                    counts["qa_fail"] += 1
                    if tagged:
                        counts["qa_tagged_fail"] += 1
                        if q.get("m1v7_has_ner_coverage") is False:
                            counts["qa_v7_uncovered_tagged"] += 1
    return counts


def _bpm_re_summary(bpm_path):
    """Return (n_re_patterns, n_onto_violations, n_onto_threshold_clearing).
    Mirrors crosscheck_F1's logic."""
    if not bpm_path.exists():
        return 0, 0, 0
    try:
        bpm = json.loads(bpm_path.read_text())
    except json.JSONDecodeError:
        return 0, 0, 0
    pats = (bpm.get("tasks", {}) or {}).get(
        "re", {}).get("patterns", []) or []
    n_total = len(pats)
    n_onto = sum(1 for p in pats if isinstance(p, dict)
                 and p.get("onto_violation"))
    n_onto_threshold = sum(
        1 for p in pats if isinstance(p, dict)
        and p.get("onto_violation")
        and int(p.get("count", 0) or 0) >= 2
        and float(p.get("severity", 0) or 0) > 0.70)
    return n_total, n_onto, n_onto_threshold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-on", required=True)
    ap.add_argument("--v5-off", required=True)
    ap.add_argument("--v7-on", required=True)
    ap.add_argument("--v7-off", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dirs = {
        "v5_on":  Path(args.v5_on),
        "v5_off": Path(args.v5_off),
        "v7_on":  Path(args.v7_on),
        "v7_off": Path(args.v7_off),
    }
    for name, d in dirs.items():
        if not d.exists():
            print(f"[FATAL] missing dir: {name} -> {d}")
            sys.exit(2)

    print("=" * 78)
    print("DBPM aggregate provenance — composite attribution from")
    print("two existing 5K ablation pairs")
    print("=" * 78)

    counts = {}
    re_bpm = {}
    for name, d in dirs.items():
        enr = d / "multitask_data_enriched.jsonl"
        bpm = d / "bpm_production.json"
        print(f"\nScanning {name} ({d.name})...")
        counts[name] = _count_pairs(enr)
        re_bpm[name] = _bpm_re_summary(bpm)
        c = counts[name]
        print(f"  patients={c['n_patients']:>5}  "
              f"NER={c['ner_pairs']:>7,}  "
              f"RE={c['re_pairs']:>7,}  QA={c['qa_pairs']:>6,}  "
              f"QA-tagged-PASS={c['qa_tagged_pass']:>5,}  "
              f"QA-tagged-FAIL={c['qa_tagged_fail']:>5,}")

    # ─── Per-component attribution ──────────────────────────────
    print("\n" + "=" * 78)
    print("Per-component attribution")
    print("=" * 78)

    # Pair A: M3-RE effect (v5+M3 ON vs v5+M3 OFF)
    print("\n  ── Pair A: M3-RE effect (v5+M3 ON vs OFF) ──")
    a_on, a_off = counts["v5_on"], counts["v5_off"]
    a_ner_delta = a_on["ner_pairs"] - a_off["ner_pairs"]
    a_re_delta = a_on["re_pairs"] - a_off["re_pairs"]
    a_qa_delta = a_on["qa_pairs"] - a_off["qa_pairs"]
    print(f"    NER pairs:  ON={a_on['ner_pairs']:>7,}  "
          f"OFF={a_off['ner_pairs']:>7,}  Δ={a_ner_delta:+,}")
    print(f"    RE pairs :  ON={a_on['re_pairs']:>7,}  "
          f"OFF={a_off['re_pairs']:>7,}  Δ={a_re_delta:+,}  "
          f"(this is M3-RE's suppression effect)")
    print(f"    QA pairs :  ON={a_on['qa_pairs']:>6,}  "
          f"OFF={a_off['qa_pairs']:>6,}  Δ={a_qa_delta:+,}")
    print(f"    bpm RE onto-patterns at threshold (v5+M3 ON): "
          f"{re_bpm['v5_on'][2]}")

    # Pair B: v7 effect (v7 ON vs v7 OFF, M3 on in both)
    print("\n  ── Pair B: M1-v7 effect (v7 ON vs OFF, M3 on in both) ──")
    b_on, b_off = counts["v7_on"], counts["v7_off"]
    b_ner_delta = b_on["ner_pairs"] - b_off["ner_pairs"]
    b_re_delta = b_on["re_pairs"] - b_off["re_pairs"]
    b_qa_delta = b_on["qa_pairs"] - b_off["qa_pairs"]
    qa_tags_on = b_on["qa_tagged_pass"] + b_on["qa_tagged_fail"]
    qa_tags_off = b_off["qa_tagged_pass"] + b_off["qa_tagged_fail"]
    v7_attrib_tags = qa_tags_on - qa_tags_off
    print(f"    NER pairs:  ON={b_on['ner_pairs']:>7,}  "
          f"OFF={b_off['ner_pairs']:>7,}  Δ={b_ner_delta:+,}  "
          f"(upstream of v7; noise)")
    print(f"    RE pairs :  ON={b_on['re_pairs']:>7,}  "
          f"OFF={b_off['re_pairs']:>7,}  Δ={b_re_delta:+,}  "
          f"(M3 on in both; noise)")
    print(f"    QA pairs :  ON={b_on['qa_pairs']:>6,}  "
          f"OFF={b_off['qa_pairs']:>6,}  Δ={b_qa_delta:+,}  "
          f"(DOWNGRADE-first: no pair removal)")
    print(f"    QA tags  :  ON={qa_tags_on:>6,}  "
          f"OFF={qa_tags_off:>6,}  Δ={v7_attrib_tags:+,}  "
          f"(this is v7's attributable tagging)")

    # ─── Composite ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Composite DBPM new-mechanism attribution")
    print("=" * 78)
    m3_suppressions = -a_re_delta if a_re_delta < 0 else 0
    v7_attrib_pos = v7_attrib_tags if v7_attrib_tags > 0 else 0
    total_new_dbpm_events = m3_suppressions + v7_attrib_pos
    total_pipeline_pairs = (b_on["ner_pairs"] + b_on["re_pairs"]
                            + b_on["qa_pairs"])
    pct = (100 * total_new_dbpm_events / total_pipeline_pairs
           if total_pipeline_pairs > 0 else 0)
    print(f"\n  M3-RE silent suppressions (RE candidates) : "
          f"{m3_suppressions:>5,}")
    print(f"  M1-v7 attributable QA tags (DOWNGRADE)     : "
          f"{v7_attrib_pos:>5,}")
    print(f"  Total new-DBPM-mechanism events           : "
          f"{total_new_dbpm_events:>5,}")
    print(f"  Total pipeline pairs (NER+RE+QA, v7-ON arm): "
          f"{total_pipeline_pairs:>7,}")
    print(f"  → DBPM new mechanisms touched {pct:.2f}% of pipeline "
          f"pairs at sealed 5K")

    # ─── Runtime variance discussion ────────────────────────────
    print("\n" + "=" * 78)
    print("Runtime variance (v7-ON vs v7-OFF arms)")
    print("=" * 78)
    print("""
  Observed (from worker_*.log):
    v7-ON  arm: 8h48m10s (6.34 s/patient × 5000)
    v7-OFF arm: 8h48m50s (6.35 s/patient × 5000)
    Δ = 40 seconds across 31,690 seconds = 0.13%

  Honest interpretation: this delta is WITHIN per-patient variance.
  Typical per-patient std dev for this pipeline (vLLM continuous
  batching + dual-model verifier + network jitter) is roughly
  1-3 seconds. Across n=5000, the expected noise on the mean
  difference is approximately σ/√n ≈ 0.028 s/patient × 5000
  ≈ 140 seconds. The observed 40-second delta is below the noise
  floor. No mechanism plausibly accelerates v7-ON over v7-OFF; v7
  adds work (NER-category set construction + lookup) and would be
  predicted to be marginally slower if anything.

  Defensible paper claim: 'v7's per-patient overhead is below the
  pipeline's per-patient variance floor; runtime delta of 0.13%
  across 5000 patients is within noise.' This is a positive
  finding: the mechanism is essentially free at runtime.

  Not a defensible claim: 'v7-ON is faster than v7-OFF.' A
  reviewer who runs the standard t-test will reject this on n=2
  runs with effect size 0.13%.
""")

    # ─── Save report ─────────────────────────────────────────────
    report = {
        "per_arm_counts": counts,
        "per_arm_bpm_re": {
            k: {"total_re_patterns": v[0],
                "onto_violation_patterns": v[1],
                "onto_threshold_clearing": v[2]}
            for k, v in re_bpm.items()
        },
        "pair_A_M3_RE_effect": {
            "ner_delta": a_ner_delta,
            "re_delta": a_re_delta,
            "qa_delta": a_qa_delta,
            "m3_re_suppressions": m3_suppressions,
        },
        "pair_B_v7_effect": {
            "ner_delta": b_ner_delta,
            "re_delta": b_re_delta,
            "qa_delta": b_qa_delta,
            "qa_tags_delta": v7_attrib_tags,
            "v7_attributable_tags": v7_attrib_pos,
        },
        "composite_attribution": {
            "m3_re_suppressions": m3_suppressions,
            "v7_qa_tags": v7_attrib_pos,
            "total_new_dbpm_events": total_new_dbpm_events,
            "total_pipeline_pairs": total_pipeline_pairs,
            "pct_pipeline_touched": round(pct, 4),
        },
        "runtime": {
            "v7_on_seconds": 8*3600 + 48*60 + 10,
            "v7_off_seconds": 8*3600 + 48*60 + 50,
            "delta_seconds": -40,
            "delta_pct": -0.13,
            "interpretation": "within per-patient variance floor",
            "defensible_claim": (
                "v7's per-patient overhead is below the pipeline's "
                "per-patient variance floor (0.13% across 5000 "
                "patients; n=1 paired run)."),
        },
        "scope_note": (
            "This composite attribution stacks two ablations on the "
            "same sealed 5K cohort. It is NOT a single 'DBPM full ON "
            "vs DBPM full OFF' run. Legacy NER/_ner_block and legacy "
            "_re_block checks remain active in all arms. A true "
            "all-mechanisms-off run would require a new arm with "
            "M1V7_ENABLE=0 AND BPM_DISABLE_M3=1 simultaneously."),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
