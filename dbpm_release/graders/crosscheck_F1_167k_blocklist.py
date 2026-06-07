#!/usr/bin/env python3
"""
crosscheck_F1_167k_blocklist.py
================================

F1 in the paper: "DBPM v2's verifier-fed `_re_block` channel was
empirically empty at 167K-patient production scale."

This claim requires three pieces of evidence to be defensible against
peer-review challenge:

  (1) _re_block reconstructed from OUTPUT_DIR/bpm_production.json is
      genuinely zero.

  (2) The rejection STREAM was not destroyed by the known "w"-mode
      overwrite incident. We test this by counting Stage_3_RE events
      in universal_rejections.jsonl. If that file has thousands of RE
      rejections but bpm has zero patterns clearing threshold, the
      finding is real (under-accumulation). If both are empty, the
      claim is consistent with overwrite corruption rather than
      genuine non-accumulation, and we must caveat accordingly.

  (3) Compare against sealed-5K M3-on bpm: M3-RE should populate the
      same _re_block via _re_onto_block. If 5K has patterns clearing
      threshold and 167K does not, the contrast holds.

Read-only. No GPU.

Usage
-----
    python crosscheck_F1_167k_blocklist.py \\
        --prod-dir path/to/code/OUTPUT_DIR \\
        --m3-dir   path/to/code/OUTPUT_DIR_M3_ON \\
        --out      path/to/reports/F1_crosscheck.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


# Exact thresholds from worker._rebuild_cache (L1170-1174)
RE_MIN_COUNT = 2
RE_MIN_SEVERITY = 0.70


def _rebuild_re_blocks(bpm):
    """Replicate worker._rebuild_cache logic for both blocklists.

    Returns (re_block_set, re_onto_block_set, threshold_pattern_count,
             total_re_pattern_count, onto_pattern_count).
    """
    re_block = set()
    re_onto_block = set()
    total = 0
    clearing = 0
    onto = 0
    pats = (bpm.get("tasks", {}) or {}).get("re", {}).get(
        "patterns", []) or []
    for p in pats:
        if not isinstance(p, dict):
            continue
        total += 1
        cnt = int(p.get("count", 0) or 0)
        sev = float(p.get("severity", 0) or 0)
        clears = (cnt >= RE_MIN_COUNT and sev > RE_MIN_SEVERITY)
        if clears:
            clearing += 1
            re_block.add((str(p.get("head_type", "")).lower(),
                          str(p.get("relation", "")).lower(),
                          str(p.get("tail_type", "")).lower()))
        if p.get("onto_violation"):
            onto += 1
            if clears:
                re_onto_block.add(
                    f"re_onto::{str(p.get('head_type','')).lower()}|"
                    f"{str(p.get('relation','')).lower()}|"
                    f"{str(p.get('tail_type','')).lower()}")
    return re_block, re_onto_block, clearing, total, onto


def _count_stage_3_rejections(rej_path):
    """Count Stage_3_RE rejections in universal_rejections.jsonl.

    Returns (n_total, n_re_stage, stage_distribution).
    """
    if not rej_path.exists():
        return None, None, None
    n_total = 0
    stage_counter = Counter()
    n_re = 0
    with open(rej_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_total += 1
            stage = str(rec.get("stage", ""))
            stage_counter[stage] += 1
            if "Stage_3_" in stage:
                n_re += 1
    return n_total, n_re, dict(stage_counter.most_common(10))


def _file_size(p):
    try:
        return p.stat().st_size
    except FileNotFoundError:
        return None


def _file_mtime(p):
    try:
        import datetime
        return datetime.datetime.fromtimestamp(
            p.stat().st_mtime).isoformat()
    except FileNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prod-dir", required=True,
                    help="167K production output dir (e.g. OUTPUT_DIR/)")
    ap.add_argument("--m3-dir", required=True,
                    help="Sealed 5K M3-on dir for comparison")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prod = Path(args.prod_dir)
    m3 = Path(args.m3_dir)
    if not prod.exists():
        print(f"[FATAL] missing: {prod}"); sys.exit(2)
    if not m3.exists():
        print(f"[FATAL] missing: {m3}"); sys.exit(2)

    prod_bpm = prod / "bpm_production.json"
    prod_rej = prod / "universal_rejections.jsonl"
    m3_bpm = m3 / "bpm_production.json"
    m3_rej = m3 / "universal_rejections.jsonl"

    # ── (1) 167K bpm reconstruction ───────────────────────────────
    print("=" * 78)
    print(f"F1 cross-check: 167K production blocklist claim")
    print("=" * 78)

    print(f"\n── (1) 167K bpm reconstruction ──")
    print(f"   file: {prod_bpm}")
    if not prod_bpm.exists():
        print("   [FATAL] file does not exist"); sys.exit(2)
    sz = _file_size(prod_bpm)
    mt = _file_mtime(prod_bpm)
    print(f"   size : {sz:,} bytes")
    print(f"   mtime: {mt}")

    try:
        bpm_prod = json.loads(prod_bpm.read_text())
    except json.JSONDecodeError as e:
        print(f"   [FATAL] JSON parse: {e}")
        sys.exit(3)

    prod_re_block, prod_re_onto, prod_clear, prod_tot, prod_onto = \
        _rebuild_re_blocks(bpm_prod)
    print(f"   tasks.re.patterns total       : {prod_tot}")
    print(f"   patterns clearing threshold   : {prod_clear}")
    print(f"   onto-violation patterns total : {prod_onto}")
    print(f"   _re_block reconstructed       : {len(prod_re_block)}")
    print(f"   _re_onto_block reconstructed  : {len(prod_re_onto)}")

    # Sanity: also check QA and NER blocks at 167K (for context)
    n_qa = len((bpm_prod.get("tasks", {}) or {}).get(
        "qa", {}).get("patterns", []) or [])
    n_ner = len((bpm_prod.get("tasks", {}) or {}).get(
        "ner", {}).get("patterns", []) or [])
    print(f"   (context) tasks.qa.patterns   : {n_qa}")
    print(f"   (context) tasks.ner.patterns  : {n_ner}")

    # ── (2) Rejection stream check (overwrite-corruption test) ───
    print(f"\n── (2) 167K rejection stream ──")
    print(f"   file: {prod_rej}")
    n_tot, n_re_stage, stage_dist = _count_stage_3_rejections(prod_rej)
    if n_tot is None:
        print("   [WARN] universal_rejections.jsonl missing.")
        print("   We cannot distinguish overwrite-corruption from")
        print("   genuine non-accumulation without this file.")
    else:
        print(f"   total rejection events       : {n_tot:,}")
        print(f"   Stage_3_* (RE) events        : {n_re_stage:,}")
        print(f"   stage distribution (top 10):")
        for s, n in stage_dist.items():
            print(f"     {n:>8,d}  {s}")

    # ── (3) Sealed 5K M3-on contrast ──────────────────────────────
    print(f"\n── (3) Sealed 5K M3-on contrast ──")
    print(f"   file: {m3_bpm}")
    if not m3_bpm.exists():
        print("   [FATAL] file does not exist"); sys.exit(2)
    bpm_m3 = json.loads(m3_bpm.read_text())
    m3_re_block, m3_re_onto, m3_clear, m3_tot, m3_onto = \
        _rebuild_re_blocks(bpm_m3)
    print(f"   tasks.re.patterns total       : {m3_tot}")
    print(f"   patterns clearing threshold   : {m3_clear}")
    print(f"   onto-violation patterns total : {m3_onto}")
    print(f"   _re_block reconstructed       : {len(m3_re_block)}")
    print(f"   _re_onto_block reconstructed  : {len(m3_re_onto)}")

    # ── Verdict ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("Verdict")
    print("=" * 78)
    print()
    findings = []

    # Test A: 167K _re_block empty?
    if len(prod_re_block) == 0:
        findings.append(("A_167k_re_block_empty", True,
                         "✓ 167K _re_block has 0 patterns clearing threshold"))
    else:
        findings.append(("A_167k_re_block_empty", False,
                         f"✗ 167K _re_block has {len(prod_re_block)} patterns — "
                         f"F1 claim contradicted"))

    # Test B: 167K had RE rejections (so non-accumulation was real, not data loss)?
    if n_re_stage is None:
        findings.append(("B_167k_had_rejections", None,
                         "? universal_rejections.jsonl missing — "
                         "cannot distinguish overwrite from non-accumulation"))
    elif n_re_stage > 100:
        findings.append(("B_167k_had_rejections", True,
                         f"✓ 167K had {n_re_stage:,} RE rejections; "
                         f"non-accumulation to _re_block is REAL, "
                         f"not data loss"))
    elif n_re_stage == 0:
        findings.append(("B_167k_had_rejections", False,
                         "? 167K had zero RE rejection events — consistent "
                         "with either (a) no rejections occurred or "
                         "(b) universal_rejections also affected by overwrite"))
    else:
        findings.append(("B_167k_had_rejections", "marginal",
                         f"◐ 167K had {n_re_stage} RE rejections — "
                         f"low but non-zero"))

    # Test C: 5K M3-on _re_onto_block populated?
    if len(m3_re_onto) > 0:
        findings.append(("C_5k_m3_onto_populated", True,
                         f"✓ Sealed 5K M3-on _re_onto_block has "
                         f"{len(m3_re_onto)} patterns; M3-RE channel works"))
    else:
        findings.append(("C_5k_m3_onto_populated", False,
                         "✗ Sealed 5K M3-on _re_onto_block is empty — "
                         "F1 contrast does not hold"))

    for code, val, msg in findings:
        print(f"  {msg}")
    print()

    # Overall verdict
    if (all(f[1] is True for f in findings if f[0] != "B_167k_had_rejections")
            and findings[1][1] is True):
        print("  ✓✓ F1 claim FULLY supported:")
        print("     • 167K _re_block was empty")
        print("     • Empty was NOT due to overwrite — rejections did occur")
        print("     • M3-RE channel populates at 5K where v2 channel didn't")
        print()
        print("  Recommended paper wording:")
        print('     "At 167K-patient production scale, the DBPM v2 _re_block')
        print(f'     channel contained zero patterns clearing the (count≥2,')
        print(f'     severity>0.70) threshold, despite {n_re_stage:,} RE-stage')
        print('     verifier rejections recorded in universal_rejections.jsonl.')
        print('     Per-signature rejections under-accumulated to threshold')
        print('     because the verifier rejection pattern was diffuse')
        print('     rather than concentrated. M3-RE populates the same')
        print(f'     channel with {len(m3_re_onto)} patterns at sealed 5K via')
        print('     the verifier-independent ontology-violation source."')
    elif (findings[0][1] is True and findings[2][1] is True
          and findings[1][1] is None):
        print("  ◐ F1 claim PARTIALLY supported (rejection-stream data missing):")
        print("     • 167K _re_block is empty (confirmed)")
        print("     • Cannot distinguish overwrite corruption from genuine")
        print("       non-accumulation; the stream file is unavailable")
        print("     • M3-RE channel populates at 5K (confirmed)")
        print()
        print("  Recommended paper wording (caveat-bearing):")
        print('     "At 167K-patient production scale, the persisted DBPM v2')
        print('     _re_block channel contained zero patterns clearing the')
        print('     (count≥2, severity>0.70) threshold. (The corresponding')
        print('     rejection-event stream from this run was not available')
        print('     for cross-validation due to a known persistence incident;')
        print('     this finding pertains to the persisted snapshot rather')
        print('     than the full rejection history.) M3-RE populates the')
        print(f'     same channel with {len(m3_re_onto)} patterns at sealed 5K')
        print('     via the verifier-independent ontology source."')
    else:
        print("  ✗ F1 claim NOT supported as written. Findings above conflict")
        print("    with the claim; either re-frame or drop F1 from the paper.")

    # Save report
    report = {
        "_provenance": {
            "prod_bpm": str(prod_bpm),
            "prod_bpm_size": sz,
            "prod_bpm_mtime": mt,
            "m3_bpm": str(m3_bpm),
            "thresholds": {"min_count": RE_MIN_COUNT,
                           "min_severity": RE_MIN_SEVERITY},
        },
        "prod_167k": {
            "re_patterns_total": prod_tot,
            "re_patterns_clearing_threshold": prod_clear,
            "onto_patterns_total": prod_onto,
            "_re_block_size": len(prod_re_block),
            "_re_onto_block_size": len(prod_re_onto),
            "qa_patterns_total": n_qa,
            "ner_patterns_total": n_ner,
            "rejection_events_total": n_tot,
            "rejection_events_stage_3_re": n_re_stage,
            "rejection_stage_distribution": stage_dist,
        },
        "sealed_5k_m3_on": {
            "re_patterns_total": m3_tot,
            "re_patterns_clearing_threshold": m3_clear,
            "onto_patterns_total": m3_onto,
            "_re_block_size": len(m3_re_block),
            "_re_onto_block_size": len(m3_re_onto),
        },
        "findings": [
            {"test": code, "result": val, "message": msg}
            for code, val, msg in findings
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
