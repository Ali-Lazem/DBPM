"""
dbpm.py — Dynamic Bidirectional Pattern Memory (DBPM)
=====================================================

Reference implementation of the DBPM gating module described in:

    [Paper 1 title], [authors], [venue], [year].  (preprint: [arXiv DOI])

DBPM is an inference-time, gradient-free "cognitive memory" that gates the
outputs of a verifier-grounded clinical extraction pipeline. It records, per
task, both FAILURES (a blocklist) and SUCCESSES (a whitelist), propagates
signal across tasks, weights evidence by source, and forgets via real-time
decay. The gate consults this memory to BLOCK, DOWNGRADE, or ALLOW each
candidate without any gradient update.

What this module IS
-------------------
The full DBPM gating logic for all seven task channels (NER, RE, QA, summary,
medications, temporal_events, plus a belief-update channel), with the
bidirectional fail/success learning, cross-task propagation, three-tier
gating, Bayesian source weighting, and time-decay that constitute the paper's
contribution. It also includes the two verifier-independent predicates used by
the QA gate (M1) and the RE ontology channel (M3).

What this module is NOT
-----------------------
This is the gating module only. The surrounding extraction pipeline
(orchestration, vLLM serving, the five-stage NER/RE/QA/summary loop, the
intelligence/risk layer, and the visualisation builder) is a separate system
and is not included here.

Dependencies
------------
Standard library only, plus `filelock` (pip install filelock) for the
merge-on-save concurrency used in multi-worker runs.

Mapping to the paper
--------------------
  * Algorithm 1 (memory update operator) ........ BadPatternMemory.record
  * Three-tier gate (BLOCK/DOWNGRADE/ALLOW) ...... gate_ner / gate_relation / gate_qa
  * Bayesian source weighting (Eq. source-weight)  SOURCE_WEIGHTS + record()
  * Real-time decay (Eq. half-life) .............. _time_decay
  * Cross-task propagation ....................... record() (is_cross_task path)
  * M1 QA error predicate ........................ m1_classify_qa_error
  * M3 RE ontology-violation predicate ........... m3_ontology_violation
"""

from __future__ import annotations

import os
import re
import json
import datetime
from filelock import FileLock


# =====================================================================
# Ontology configuration consumed by the gate predicates.
# (Extracted from the pipeline's shared ontology; required by the M3-RE
#  predicate and the RE whitelist. Edit to match your own task schema.)
# =====================================================================

# Entity-type-pair -> allowed relation labels the worker may PROPOSE.
ENTITY_RELATION_PRIORS = {
    ("Treatment", "Diagnosis"): ["TREATS", "CAUSES", "PREVENTS", "WORSENS", "ASSOCIATED_WITH"],
    ("Treatment", "Condition"): ["TREATS", "CAUSES", "PREVENTS", "WORSENS"],
    ("Treatment", "Symptom"):   ["TREATS", "CAUSES", "ALLEVIATES", "SIDE_EFFECT_OF"],
    ("Treatment", "Symptoms"):  ["TREATS", "CAUSES", "ALLEVIATES", "SIDE_EFFECT_OF"],
    ("Treatment", "Test"):      ["NECESSITATES", "GUIDED_BY"],
    ("Treatment", "Tests"):     ["NECESSITATES", "GUIDED_BY"],
    ("Procedure", "Diagnosis"): ["TREATS", "DIAGNOSES", "MANAGED"],
    ("Procedure", "Condition"): ["TREATS", "DIAGNOSES", "MANAGED"],
    ("Medication", "Diagnosis"): ["TREATS", "CAUSES", "PREVENTS"],
    ("Medication", "Symptom"):   ["TREATS", "CAUSES", "SIDE_EFFECT_OF"],
    ("Diagnosis", "Symptom"):   ["MANIFESTS_AS", "CAUSES", "ASSOCIATED_WITH"],
    ("Condition", "Symptom"):   ["MANIFESTS_AS", "CAUSES"],
    ("Symptom", "Diagnosis"):   ["INDICATES", "SUGGESTS", "ASSOCIATED_WITH"],
    ("Symptoms", "Diagnosis"):  ["INDICATES", "SUGGESTS", "ASSOCIATED_WITH"],
    ("Symptoms", "Symptoms"):   ["CAUSES", "EXACERBATES", "ASSOCIATED_WITH"],
    ("Diagnosis", "Diagnosis"): ["CAUSES", "COMPLICATED_BY", "ASSOCIATED_WITH"],
    ("Condition", "Condition"): ["CAUSES", "COMPLICATED_BY"],
    ("Test", "Diagnosis"):       ["REVEALS", "RULES_OUT", "CONFIRMS"],
    ("Tests", "Diagnosis"):      ["REVEALS", "RULES_OUT", "CONFIRMS"],
    ("Test", "Symptom"):         ["REVEALS", "EXPLAINS"],
    ("Tests", "Symptom"):        ["REVEALS", "EXPLAINS"],
    ("Risk_Factor", "Diagnosis"): ["INCREASES_RISK_OF", "CAUSES"],
    ("Risk_Factor", "Condition"): ["INCREASES_RISK_OF", "CAUSES"],
    ("Diagnosis", "Outcome"):     ["LEADS_TO", "CAUSES", "RESULTED_IN"],
    ("Diagnosis", "Outcome_Mortality"): ["CAUSES", "RESULTED_IN"],
    ("Treatment", "Outcome"):     ["RESULTS_IN", "FAILED_TO_PREVENT"],
    ("History", "Diagnosis"):    ["HISTORY_OF", "PRECEDES"],
    ("History", "Condition"):    ["HISTORY_OF", "PRECEDES"],
    ("Diagnosis", "History"):    ["FOLLOWS"],
}

# Surface-form normalisation: maps many relation spellings onto one vocabulary.
RELATION_ALIAS = {
    "ALLEVIATES": "TREATS", "MANAGED": "TREATS", "MANAGED_BY": "TREATS",
    "RESOLVES": "TREATS", "SIDE_EFFECT_OF": "CAUSES", "WORSENS": "CAUSES",
    "EXACERBATES": "CAUSES", "RESULTS_IN": "CAUSES", "LEADS_TO": "CAUSES",
    "RESULTED_IN": "CAUSES", "INCREASES_RISK_OF": "CAUSES", "INDUCED_BY": "CAUSES",
    "DUE_TO": "CAUSES", "COMPLICATED_BY": "COMPLICATED_BY",
    "MANIFESTS_AS": "MANIFESTS", "INDICATES": "REVEALS", "SUGGESTS": "REVEALS",
    "CONFIRMS": "REVEALS", "RULES_OUT": "REVEALS", "EXPLAINS": "REVEALS",
    "DIAGNOSES": "REVEALS", "GUIDED_BY": "REVEALS", "PRECEDES": "BEFORE",
    "HISTORY_OF": "BEFORE", "FOLLOWS": "AFTER", "NECESSITATES": "ASSOCIATED_WITH",
    "PREVENTS": "PREVENTS", "TREATS": "TREATS", "CAUSES": "CAUSES",
    "REVEALS": "REVEALS", "MANIFESTS": "MANIFESTS", "ASSOCIATED_WITH": "ASSOCIATED_WITH",
}


# =====================================================================
# Verifier-independent predicates
# =====================================================================

def _norm(s):
    return str(s).strip().upper()


def m3_ontology_violation(head_type, relation, tail_type):
    """M3-RE predicate (verifier-independent, deterministic).

    True iff (head_type, tail_type) IS a known ENTITY_RELATION_PRIORS pair
    AND the alias-normalised relation is NOT in that pair's allowed set
    (i.e. a contradiction of the ontology). Unknown pairs return False
    (benign-common, not a bad signal). No model call.
    """
    priors = ENTITY_RELATION_PRIORS
    alias = RELATION_ALIAS
    if not priors:
        return False
    h, t = _norm(head_type), _norm(tail_type)
    key = (h, t)
    if key not in {(_norm(a), _norm(b)) for (a, b) in priors}:
        return False
    allowed = set()
    for (a, b), rels in priors.items():
        if (_norm(a), _norm(b)) == key:
            for rr in rels:
                allowed.add(alias.get(rr.upper(), rr.upper()))
    r = alias.get(_norm(relation), _norm(relation))
    return r not in allowed


def m3_onto_key(head_type, relation, tail_type):
    """String key for the disjoint RE ontology channel. A str can never
    equal the tuple keys used by the verifier-rejection blocklist, so the
    two channels are disjoint by type."""
    return f"re_onto::{_norm(head_type)}|{_norm(relation)}|{_norm(tail_type)}"


_NEG_CUES_M1 = (" no ", " not ", " denies", " denied", " without ",
                " absent", " negative for", " ruled out", " r/o ",
                " never ", " none ", " resolved", " free of ")
_NUM_RE_M1 = re.compile(r"\b\d[\d.,/:%-]*\b")


def _m1_tokset(s):
    return set(re.findall(r"[a-z]{3,}", str(s).lower()))


def m1_classify_qa_error(question, answer, ctx):
    """M1 QA error predicate (verifier-independent, pure string ops).

    Returns one of: 'unsupported' | 'contradicted' | 'over_specified'.
    A coarse 3-class taxonomy for SEPARABILITY of the QA reject signal,
    not a validated clinical error ontology. `question` is accepted for
    signature stability but unused by the 3-class logic.
    """
    a = str(answer or "").lower()
    c = str(ctx or "").lower()
    if not a.strip():
        return "unsupported"
    a_tok = _m1_tokset(a)
    if not a_tok:
        return "unsupported"
    c_tok = _m1_tokset(c)
    overlap = len(a_tok & c_tok) / max(1, len(a_tok))
    if overlap < 0.15:
        return "unsupported"
    neg_in_c = any(cue in (" " + c + " ") for cue in _NEG_CUES_M1)
    neg_in_a = any(cue in (" " + a + " ") for cue in _NEG_CUES_M1)
    if overlap >= 0.30 and (neg_in_c != neg_in_a):
        return "contradicted"
    a_nums = set(_NUM_RE_M1.findall(a))
    c_nums = set(_NUM_RE_M1.findall(c))
    if overlap >= 0.15 and a_nums and not (a_nums & c_nums):
        return "over_specified"
    return "unsupported"


def m1_assign_error_class(question, answer, ctx):
    """Resilient wrapper around m1_classify_qa_error. Logs the exception
    type and RE-RAISES under DBPM_DEV_STRICT=1 (so a broken classifier
    fails loud on dev); defaults to 'unsupported' in production."""
    try:
        return m1_classify_qa_error(question, answer, ctx)
    except Exception as e:  # pragma: no cover
        import sys
        sys.stderr.write(f"[M1][WARN] classifier raised {type(e).__name__}: {e}\n")
        if os.environ.get("DBPM_DEV_STRICT") == "1":
            raise
        return "unsupported"


# =====================================================================
# Feature flags (environment-controlled; default all ON).
# Set any to "0" to disable for ablation studies. DBPM_DISABLE=1 is the
# master switch that turns the whole memory off.
# =====================================================================
DBPM_ENABLE_CROSS_TASK     = os.environ.get("BPM_ENABLE_CROSS_TASK",     "1") == "1"
DBPM_ENABLE_UNCERTAINTY    = os.environ.get("BPM_ENABLE_UNCERTAINTY",    "1") == "1"
DBPM_ENABLE_THREE_TIER     = os.environ.get("BPM_ENABLE_THREE_TIER",     "1") == "1"
DBPM_ENABLE_SOURCE_WEIGHTS = os.environ.get("BPM_ENABLE_SOURCE_WEIGHTS", "1") == "1"
DBPM_GLOBAL_DISABLE        = os.environ.get("BPM_DISABLE",               "0") == "1"
DBPM_DISABLE_M3            = os.environ.get("BPM_DISABLE_M3",            "0") == "1"

if DBPM_GLOBAL_DISABLE:
    DBPM_ENABLE_CROSS_TASK = False
    DBPM_ENABLE_UNCERTAINTY = False
    DBPM_ENABLE_THREE_TIER = False
    DBPM_ENABLE_SOURCE_WEIGHTS = False


class BadPatternMemory:
    """Dynamic Bidirectional Pattern Memory (DBPM).

    Records FAILURES (blocklist) and SUCCESSES (whitelist) per task, with:
      1. Bidirectional learning (success whitelist + failure blocklist)
      2. Cross-task signal propagation (e.g. RE hard-fail warns NER)
      3. Severity-tier learning (hard FAIL vs soft DOWNGRADE vs SUCCESS rates)
      4. Real-time exponential forgetting (UTC-time half-life, not step count)
      5. Per-task analytics for post-run audit
      6. Coverage of all seven tasks (NER, RE, QA, summary, medications,
         temporal_events, belief_update)

    Persistence is merge-on-save under a FileLock so many workers can share
    one memory file safely.
    """

    # How long a pattern stays relevant (decay half-life, in days).
    TASK_WINDOWS = {
        "ner": 5, "re": 10, "qa": 10, "summary": 20,
        "medications": 8, "temporal_events": 15,
        "belief_update": 30, "default": 10,
    }

    # Learning rates: hard failure learns faster than soft downgrade; success slowest.
    LEARN_RATES = {
        "hard_fail":     {"ner": 0.15, "re": 0.20, "qa": 0.15, "summary": 0.30, "default": 0.18},
        "soft_downgrade": {"ner": 0.05, "re": 0.08, "qa": 0.05, "summary": 0.12, "default": 0.07},
        "success":       {"ner": 0.03, "re": 0.05, "qa": 0.04, "summary": 0.08, "default": 0.04},
    }

    # Severity a pattern must reach before it BLOCKs.
    BLOCK_THRESHOLDS = {
        "re":      {"min_count": 2, "min_severity": 0.70},
        "ner":     {"min_count": 3, "min_severity": 0.65},
        "qa":      {"min_count": 3, "min_severity": 0.60},
        "summary": {"min_count": 2, "min_severity": 0.45},
        "default": {"min_count": 3, "min_severity": 0.65},
    }

    # Three-tier gating: severities in the (downgrade, block] band DOWNGRADE.
    DOWNGRADE_THRESHOLDS = {
        "re":      {"min_count": 1, "min_severity": 0.40},
        "ner":     {"min_count": 1, "min_severity": 0.40},
        "qa":      {"min_count": 1, "min_severity": 0.40},
        "summary": {"min_count": 1, "min_severity": 0.30},
        "default": {"min_count": 1, "min_severity": 0.40},
    }

    # Per-source evidence weights (Bayesian severity update). Higher = stronger.
    SOURCE_WEIGHTS = {
        "verifier_mmed":     1.0,    # direct verifier verdict — strongest evidence
        # 'verifier_self' is RESERVED for the optional self-verification
        # ablation arm and is DISABLED in the reported configuration.
        "verifier_self":     0.7,
        "rule_based":        0.5,    # ontology gatekeeper — moderate trust
        "cross_task_re":     0.3,    # RE -> NER propagation
        "cross_task_qa":     0.2,    # QA -> NER propagation (weakest)
        "cross_task_ner":    0.4,    # NER -> RE preemption (NER is upstream)
        "cross_task":        0.5,    # generic cross-task fallback
        "default":           1.0,
    }

    MAX_PATTERNS = {
        "ner": 200, "re": 300, "qa": 150, "summary": 100,
        "medications": 100, "temporal_events": 100, "default": 150,
    }

    def __init__(self, path="bpm_production.json"):
        self.path = path
        self.data = self._load()
        self._analytics_baseline = self._snapshot_analytics()
        self._rebuild_cache()

    # ---- persistence -------------------------------------------------

    def _snapshot_analytics(self):
        snap = {}
        for task, an in self.data.get("analytics", {}).items():
            if isinstance(an, dict):
                snap[task] = {
                    "total_fails":     int(an.get("total_fails", 0)),
                    "total_successes": int(an.get("total_successes", 0)),
                    "hard_fails":      int(an.get("hard_fails", 0)),
                    "downgrades":      int(an.get("downgrades", 0)),
                }
        return snap

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    d = json.load(f)
                    d.setdefault("meta", {})
                    d.setdefault("tasks", {})
                    d.setdefault("whitelist", {})
                    d.setdefault("analytics", {})
                    return d
            except Exception:
                pass
        return {
            "meta": {"version": "2.0", "created_at": datetime.datetime.utcnow().isoformat()},
            "tasks": {}, "whitelist": {}, "analytics": {},
        }

    # ---- signatures & decay -----------------------------------------

    def _get_sig(self, task, p):
        if task == "re":
            return (f"{_norm(p.get('head_type'))}|"
                    f"{_norm(p.get('relation'))}|"
                    f"{_norm(p.get('tail_type'))}")
        elif task in ("ner", "medications"):
            return str(p.get("pattern", p.get("drug", ""))).lower().strip()
        elif task == "qa":
            # Error-discriminative key for the reject path: reject patterns
            # carry error_class (disjoint from the success/whitelist space).
            ec = p.get("error_class")
            if ec:
                cat = str(p.get("category", "") or "Other").strip()
                return f"qa::{cat}|{str(ec).strip()}".lower()
            return str(p.get("question", "")).lower().strip()
        else:
            return str(sorted((k, v) for k, v in p.items()
                              if k not in {"count", "severity", "last_seen_step",
                                           "last_seen_time", "created_at"}))

    def _time_decay(self, severity, last_seen_iso, task):
        """Real-time decay: severity halves every TASK_WINDOW days."""
        try:
            last = datetime.datetime.fromisoformat(last_seen_iso)
            days = (datetime.datetime.utcnow() - last).total_seconds() / 86400
            window = self.TASK_WINDOWS.get(task, self.TASK_WINDOWS["default"])
            factor = 0.5 ** (days / window)
            return max(0.0, severity * factor)
        except Exception:
            return severity

    # ---- the memory update operator (Algorithm 1) -------------------

    def record(self, task, pattern, failure_type="hard_fail", is_cross_task=False):
        """Record a pattern observation.

        failure_type: 'hard_fail' | 'soft_downgrade' | 'success'
        Routes successes to the whitelist and failures to the blocklist,
        applies source-weighted Bayesian severity updates, and (for hard
        failures) propagates signal across tasks when cross-task is enabled.
        """
        if not hasattr(self, "_event_counters"):
            self._event_counters = {
                "ner_block": 0, "ner_success": 0, "ner_downgrade": 0,
                "re_block": 0, "re_success": 0, "re_downgrade": 0,
                "qa_block": 0, "qa_success": 0,
                "summary_fail": 0, "cross_task_propagations": 0,
            }
        if failure_type == "success":
            ck = f"{task}_success"
        elif failure_type == "soft_downgrade":
            ck = f"{task}_downgrade"
        else:
            ck = f"{task}_block"
        if ck in self._event_counters:
            self._event_counters[ck] += 1

        if "relation" in pattern:
            pattern["relation"] = _norm(pattern["relation"])
        sig = self._get_sig(task, pattern)
        key = f"{task}:{sig}"
        now = datetime.datetime.utcnow().isoformat()

        if failure_type == "success":
            wl = self.data.setdefault("whitelist", {}).setdefault(task, {})
            if sig not in wl:
                wl[sig] = {"count": 1, "last_seen": now, "confidence": 0.90,
                           "category": pattern.get("category", "")}
            else:
                wl[sig]["count"] += 1
                wl[sig]["last_seen"] = now
                wl[sig]["confidence"] = min(
                    0.99, wl[sig]["confidence"] + self.LEARN_RATES["success"].get(task, 0.04))
                if not wl[sig].get("category"):
                    wl[sig]["category"] = pattern.get("category", "")
            if not hasattr(self, "_wl_cache"):
                self._wl_cache = {}
            self._wl_cache.setdefault(task, set()).add(sig)
        else:
            rate = self.LEARN_RATES[failure_type].get(
                task, self.LEARN_RATES[failure_type]["default"])
            source_weight = 1.0
            if DBPM_ENABLE_SOURCE_WEIGHTS:
                source = pattern.get("evidence_source", "verifier_mmed")
                source_weight = self.SOURCE_WEIGHTS.get(source, self.SOURCE_WEIGHTS["default"])
                rate = rate * source_weight

            existing = getattr(self, "_write_index", {}).get(key)
            if existing:
                existing["count"] = existing.get("count", 0) + 1
                existing["last_seen_time"] = now
                decay_rate = {"hard_fail": 0.99, "soft_downgrade": 0.985,
                              "success": 0.998}.get(failure_type, 0.995)
                decayed = existing.get("severity", 0.3) * decay_rate
                existing["severity"] = min(1.0, decayed + rate)
                existing["failure_type"] = failure_type
            else:
                init_sev = 0.50 if failure_type == "hard_fail" else 0.25
                if DBPM_ENABLE_SOURCE_WEIGHTS:
                    init_sev = init_sev * source_weight
                pattern.update({"count": 1, "last_seen_time": now, "severity": init_sev,
                                "failure_type": failure_type, "created_at": now})
                target = self.data["tasks"].setdefault(task, {"patterns": []})["patterns"]
                target.append(pattern)
                if not hasattr(self, "_write_index"):
                    self._write_index = {}
                self._write_index[key] = pattern

            # Cross-task propagation (only on direct hard failures).
            if is_cross_task:
                pass
            elif DBPM_ENABLE_CROSS_TASK and failure_type == "hard_fail":
                if task == "re":
                    ner_wl = self._wl_cache.get("ner", set()) if hasattr(self, "_wl_cache") else set()
                    for ent_field in ("head", "tail"):
                        ent_str = pattern.get(ent_field, "")
                        if not (ent_str and 4 < len(ent_str.strip()) < 60):
                            continue
                        en = ent_str.lower().strip()
                        if en in ner_wl:
                            continue
                        nk = f"ner:{en}"
                        ex = self._write_index.get(nk, {})
                        if ex.get("cross_task_hits", 0) >= 3 or ex.get("severity", 0) >= 0.55:
                            continue
                        self._record_cross_task("ner", {"pattern": en, "evidence_source": "cross_task_re"})
                        up = self._write_index.get(nk)
                        if up is not None:
                            up["cross_task_hits"] = up.get("cross_task_hits", 0) + 1
                        self._event_counters["cross_task_propagations"] = \
                            self._event_counters.get("cross_task_propagations", 0) + 1
                elif task == "ner":
                    ep = pattern.get("pattern", "")
                    if ep and len(ep.strip()) > 4:
                        if not hasattr(self, "_re_blocked_ents_set"):
                            eb = self.data.get("blocked_entities_for_re", [])
                            self._re_blocked_ents_set = set(eb) if isinstance(eb, list) else eb
                        en = ep.lower().strip()
                        if en not in self._re_blocked_ents_set:
                            self._re_blocked_ents_set.add(en)
                            self.data["blocked_entities_for_re"] = list(self._re_blocked_ents_set)
                            self._event_counters["cross_task_propagations"] = \
                                self._event_counters.get("cross_task_propagations", 0) + 1
                elif task == "qa":
                    ans = pattern.get("answer", "")
                    if ans:
                        ner_wl = self._wl_cache.get("ner", set()) if hasattr(self, "_wl_cache") else set()
                        for sub in ans.split(","):
                            sub = sub.strip().lower()
                            if not (4 < len(sub) < 60) or sub in ner_wl:
                                continue
                            nk = f"ner:{sub}"
                            ex = self._write_index.get(nk, {})
                            if ex.get("cross_task_hits", 0) >= 3 or ex.get("severity", 0) >= 0.55:
                                continue
                            self._record_cross_task("ner", {"pattern": sub, "evidence_source": "cross_task_qa"})
                            up = self._write_index.get(nk)
                            if up is not None:
                                up["cross_task_hits"] = up.get("cross_task_hits", 0) + 1
                            self._event_counters["cross_task_propagations"] = \
                                self._event_counters.get("cross_task_propagations", 0) + 1

        an = self.data.setdefault("analytics", {}).setdefault(
            task, {"total_fails": 0, "total_successes": 0, "hard_fails": 0, "downgrades": 0})
        if failure_type == "success":
            an["total_successes"] = an.get("total_successes", 0) + 1
        elif failure_type == "hard_fail":
            an["hard_fails"] = an.get("hard_fails", 0) + 1
            an["total_fails"] = an.get("total_fails", 0) + 1
        else:
            an["downgrades"] = an.get("downgrades", 0) + 1
            an["total_fails"] = an.get("total_fails", 0) + 1

        self._unsaved_changes = getattr(self, "_unsaved_changes", 0) + 1
        if self._unsaved_changes >= 200:
            self.save()
            self._unsaved_changes = 0

    def _record_cross_task(self, task, pattern):
        """Cross-task recorder at half strength. Sets is_cross_task=True to
        prevent recursive propagation, then halves the severity increment."""
        sig = self._get_sig(task, pattern)
        key = f"{task}:{sig}"
        existing = getattr(self, "_write_index", {}).get(key)
        sev_before = existing.get("severity", 0.0) if existing else 0.0
        if "evidence_source" not in pattern:
            pattern["evidence_source"] = "cross_task"
        self.record(task, pattern, failure_type="soft_downgrade", is_cross_task=True)
        after = getattr(self, "_write_index", {}).get(key)
        if after:
            sev_after = after.get("severity", 0.0)
            after["severity"] = sev_before + (sev_after - sev_before) * 0.5

    def _prune(self):
        """Apply real-time decay and enforce per-task caps."""
        for task, td in self.data.get("tasks", {}).items():
            pats = td.get("patterns", [])
            for p in pats:
                p["severity"] = self._time_decay(
                    p.get("severity", 0.5), p.get("last_seen_time", "2020-01-01"), task)
                if p.get("cross_task_hits", 0) > 0:
                    p["cross_task_hits"] = max(0, int(p.get("cross_task_hits", 0) * 0.9))
            td["patterns"] = [p for p in pats if p.get("severity", 0) > 0.01]
            cap = self.MAX_PATTERNS.get(task, self.MAX_PATTERNS["default"])
            if len(td["patterns"]) > cap:
                td["patterns"].sort(key=lambda x: x.get("severity", 0))
                td["patterns"] = td["patterns"][-cap:]

    def save(self):
        """Prune, then merge-on-write under a FileLock so concurrent workers
        combine their blocklists, whitelists, and analytics safely."""
        self._prune()
        self.data["meta"]["updated_at"] = datetime.datetime.utcnow().isoformat()
        lock = self.path + ".lock"
        tmp = self.path + ".tmp"
        try:
            with FileLock(lock, timeout=600):
                if os.path.exists(self.path):
                    try:
                        with open(self.path) as f:
                            disk = json.load(f)
                        all_tasks = set(self.data["tasks"]) | set(disk.get("tasks", {}))
                        for t in all_tasks:
                            lt = {self._get_sig(t, p): p
                                  for p in self.data["tasks"].get(t, {}).get("patterns", [])}
                            for dp in disk.get("tasks", {}).get(t, {}).get("patterns", []):
                                s = self._get_sig(t, dp)
                                if s not in lt:
                                    self.data["tasks"].setdefault(t, {"patterns": []})["patterns"].append(dp)
                                else:
                                    e = lt[s]
                                    e["count"] = max(e.get("count", 0), dp.get("count", 0))
                                    e["severity"] = max(e.get("severity", 0), dp.get("severity", 0))
                        for t, wl in disk.get("whitelist", {}).items():
                            local_wl = self.data.setdefault("whitelist", {}).setdefault(t, {})
                            for s, wd in wl.items():
                                if s not in local_wl:
                                    local_wl[s] = wd
                                else:
                                    local_wl[s]["count"] = max(local_wl[s].get("count", 0), wd.get("count", 0))
                        disk_an = disk.get("analytics", {})
                        local_an = self.data.setdefault("analytics", {})
                        baseline = getattr(self, "_analytics_baseline", {})
                        merged = {}
                        for t in set(disk_an) | set(local_an) | set(baseline):
                            d_t, l_t, b_t = disk_an.get(t, {}) or {}, local_an.get(t, {}) or {}, baseline.get(t, {}) or {}
                            merged[t] = {}
                            for ctr in ("total_fails", "total_successes", "hard_fails", "downgrades"):
                                delta = int(l_t.get(ctr, 0)) - int(b_t.get(ctr, 0))
                                merged[t][ctr] = max(0, int(d_t.get(ctr, 0)) + delta)
                        self.data["analytics"] = merged
                        self._analytics_baseline = {t: dict(c) for t, c in merged.items()}
                    except Exception as me:  # pragma: no cover
                        print(f"[DBPM] Merge warn: {me}")
                with open(tmp, "w") as f:
                    json.dump(self.data, f, indent=2)
                os.replace(tmp, self.path)
            self._rebuild_cache()
        except Exception as e:  # pragma: no cover
            print(f"[DBPM] Save failed: {e}")

    # ---- O(1) cache rebuild -----------------------------------------

    def _rebuild_cache(self):
        """Build O(1) blocklist + downgrade + whitelist caches from data."""
        self._re_block = set()
        thr = self.BLOCK_THRESHOLDS.get("re", {})
        for p in self.data.get("tasks", {}).get("re", {}).get("patterns", []):
            if p.get("count", 0) >= thr.get("min_count", 2) and p.get("severity", 0) > thr.get("min_severity", 0.7):
                self._re_block.add((_norm(p.get("head_type")), _norm(p.get("relation")), _norm(p.get("tail_type"))))

        # Disjoint RE ontology-violation channel (string-keyed).
        self._re_onto_block = set()
        if not DBPM_DISABLE_M3:
            _ot = self.BLOCK_THRESHOLDS.get("re", {})
            for p in self.data.get("tasks", {}).get("re", {}).get("patterns", []):
                if not p.get("onto_violation"):
                    continue
                if p.get("count", 0) >= _ot.get("min_count", 2) and p.get("severity", 0) > _ot.get("min_severity", 0.7):
                    self._re_onto_block.add(
                        f"re_onto::{_norm(p.get('head_type'))}|{_norm(p.get('relation'))}|{_norm(p.get('tail_type'))}")

        self._ner_block = set()
        thr = self.BLOCK_THRESHOLDS.get("ner", {})
        for p in self.data.get("tasks", {}).get("ner", {}).get("patterns", []):
            if p.get("count", 0) >= thr.get("min_count", 3) and p.get("severity", 0) > thr.get("min_severity", 0.65):
                self._ner_block.add(p.get("pattern", "").lower().strip())

        self._qa_block = set()
        thr = self.BLOCK_THRESHOLDS.get("qa", {})
        for p in self.data.get("tasks", {}).get("qa", {}).get("patterns", []):
            if p.get("count", 0) >= thr.get("min_count", 3) and p.get("severity", 0) > thr.get("min_severity", 0.6):
                self._qa_block.add(p.get("question", "").lower().strip())

        # Per-category QA fail-rate signal (instance-level gate input).
        self._qa_cat_block = set()
        self._qa_cat_mass = {}
        try:
            self._qa_cat_fail_rate = self._build_qa_cat_fail_rate()
        except Exception:
            self._qa_cat_fail_rate = {}

        self._re_downgrade = set()
        bthr = self.BLOCK_THRESHOLDS.get("re", {}).get("min_severity", 0.7)
        dthr = self.DOWNGRADE_THRESHOLDS.get("re", {}).get("min_severity", 0.4)
        for p in self.data.get("tasks", {}).get("re", {}).get("patterns", []):
            sev = p.get("severity", 0)
            if dthr < sev <= bthr:
                self._re_downgrade.add((_norm(p.get("head_type")), _norm(p.get("relation")), _norm(p.get("tail_type"))))

        self._ner_downgrade = set()
        bthr = self.BLOCK_THRESHOLDS.get("ner", {}).get("min_severity", 0.65)
        dthr = self.DOWNGRADE_THRESHOLDS.get("ner", {}).get("min_severity", 0.4)
        for p in self.data.get("tasks", {}).get("ner", {}).get("patterns", []):
            sev = p.get("severity", 0)
            if dthr < sev <= bthr:
                self._ner_downgrade.add(p.get("pattern", "").lower().strip())

        self._qa_downgrade = set()
        bthr = self.BLOCK_THRESHOLDS.get("qa", {}).get("min_severity", 0.6)
        dthr = self.DOWNGRADE_THRESHOLDS.get("qa", {}).get("min_severity", 0.4)
        for p in self.data.get("tasks", {}).get("qa", {}).get("patterns", []):
            sev = p.get("severity", 0)
            if dthr < sev <= bthr:
                self._qa_downgrade.add(p.get("question", "").lower().strip())

        self._wl_cache = {}
        for t, wl in self.data.get("whitelist", {}).items():
            self._wl_cache[t] = set(wl.keys())

        sum_pats = self.data.get("tasks", {}).get("summary", {}).get("patterns", [])
        self._sum_risky = sum(1 for p in sum_pats if p.get("severity", 0) > 0.4) > 5

        # RE whitelist from ontology priors.
        self._re_whitelist = {}
        for (ph, pt), rels in ENTITY_RELATION_PRIORS.items():
            k = (_norm(ph), _norm(pt))
            self._re_whitelist[k] = {RELATION_ALIAS.get(r.upper(), r.upper()) for r in rels}

        self._write_index = {}
        for task, td in self.data.get("tasks", {}).items():
            for p in td.get("patterns", []):
                self._write_index[f"{task}:{self._get_sig(task, p)}"] = p

    # ---- O(1) gatekeepers -------------------------------------------

    def is_bad_relation(self, head_type, relation, tail_type):
        h, r, t = _norm(head_type), _norm(relation), _norm(tail_type)
        if (h, t) in self._re_whitelist and r in self._re_whitelist[(h, t)]:
            return False
        return (h, r, t) in self._re_block

    def is_blocked_entity_for_re(self, entity_text):
        blocked = self.data.get("blocked_entities_for_re", [])
        if not blocked:
            return False
        if not hasattr(self, "_re_blocked_ents_set"):
            self._re_blocked_ents_set = set(blocked) if isinstance(blocked, list) else blocked
        return entity_text.lower().strip() in self._re_blocked_ents_set

    def gate_relation(self, head_type, relation, tail_type):
        """Three-tier RE gate -> 'BLOCK' | 'DOWNGRADE' | 'ALLOW'.
        Whitelist > block > ontology channel > downgrade > allow.
        The ontology channel is DOWNGRADE-first (flag M3_RE_HARD_BLOCK=1
        promotes it to BLOCK)."""
        if not DBPM_ENABLE_THREE_TIER:
            return "BLOCK" if self.is_bad_relation(head_type, relation, tail_type) else "ALLOW"
        h, r, t = _norm(head_type), _norm(relation), _norm(tail_type)
        if (h, t) in self._re_whitelist and r in self._re_whitelist[(h, t)]:
            return "ALLOW"
        if (h, r, t) in self._re_block:
            return "BLOCK"
        if getattr(self, "_re_onto_block", None) and f"re_onto::{h}|{r}|{t}" in self._re_onto_block:
            return "BLOCK" if os.environ.get("M3_RE_HARD_BLOCK") == "1" else "DOWNGRADE"
        if (h, r, t) in self._re_downgrade:
            return "DOWNGRADE"
        return "ALLOW"

    def gate_ner(self, entity_text, category):
        """Three-tier NER gate -> 'BLOCK' | 'DOWNGRADE' | 'ALLOW'.
        Whitelist takes precedence over downgrade and block."""
        if not DBPM_ENABLE_THREE_TIER:
            return "BLOCK" if self.is_bad_ner(entity_text, category) else "ALLOW"
        norm = entity_text.lower().strip()
        if hasattr(self, "_wl_cache") and norm in self._wl_cache.get("ner", set()):
            return "ALLOW"
        if norm in self._ner_block:
            return "BLOCK"
        if norm in self._ner_downgrade:
            return "DOWNGRADE"
        return "ALLOW"

    def is_bad_ner(self, entity_text, category):
        return entity_text.lower().strip() in self._ner_block

    def gate_qa(self, category, difficulty=None, ner_categories=None):
        """Neuro-symbolic QA gate -> 'BLOCK' | 'DOWNGRADE' | 'ALLOW'.

        Layered fallbacks:
          * neuro-symbolic (when ner_categories given): the QA category must
            appear in the patient's NER category set, else DOWNGRADE.
          * category fail-rate x per-patient difficulty (when no NER signal).
          * pure category fail-rate (when difficulty disabled).
        DOWNGRADE-first; hard BLOCK only under M1_QA_GATE_HARD_BLOCK=1.
        """
        cat = str(category or "").lower().strip()
        if not cat:
            return "ALLOW"
        for tmpl in getattr(self, "_qa_whitelist_flat", []):
            if cat and (cat in tmpl or tmpl in cat):
                return "ALLOW"
        v7_enabled = os.environ.get("M1V7_ENABLE", "1") == "1"
        hard_block = os.environ.get("M1_QA_GATE_HARD_BLOCK", "0") == "1"
        no_ner_bucket = {"adverse_event", "outcome_mortality"}
        if v7_enabled and ner_categories is not None and cat not in no_ner_bucket:
            try:
                ner_set = {str(c).lower().strip() for c in ner_categories if c}
            except (TypeError, AttributeError):
                ner_set = set()
            if cat in ner_set:
                return "ALLOW"
            if not DBPM_ENABLE_THREE_TIER:
                return "BLOCK" if hard_block else "ALLOW"
            return "BLOCK" if hard_block else "DOWNGRADE"
        rate_map = getattr(self, "_qa_cat_fail_rate", None) or {}
        rate = rate_map.get(cat)
        if rate is None:
            return "ALLOW"
        try:
            rate_threshold = float(os.environ.get("M1V5_FAIL_THRESHOLD", "0.30"))
            diff_threshold = float(os.environ.get("M1V6_DIFFICULTY_THRESHOLD", "0.30"))
            v6_enabled = os.environ.get("M1V6_ENABLE", "1") == "1"
        except (ValueError, TypeError):
            rate_threshold, diff_threshold, v6_enabled = 0.30, 0.30, True
        if rate < rate_threshold:
            return "ALLOW"
        if v6_enabled and difficulty is not None:
            try:
                if float(difficulty) < diff_threshold:
                    return "ALLOW"
            except (ValueError, TypeError):
                pass
        if not DBPM_ENABLE_THREE_TIER:
            return "BLOCK" if hard_block else "ALLOW"
        return "BLOCK" if hard_block else "DOWNGRADE"

    def _build_qa_cat_fail_rate(self):
        """Per-category P(FAIL) from both surfaces: fails (tasks.qa.patterns
        with error_class) and successes (whitelist.qa with category).
        Categories with < min_n observations are excluded."""
        from collections import Counter
        min_n = 10
        totals, fails = Counter(), Counter()
        for p in self.data.get("tasks", {}).get("qa", {}).get("patterns", []):
            if not isinstance(p, dict):
                continue
            cat = str(p.get("category", "") or "").lower().strip()
            if not cat:
                continue
            n = int(p.get("count", 0) or 0)
            if n <= 0:
                continue
            totals[cat] += n
            if p.get("error_class"):
                fails[cat] += n
        for sig, val in (self.data.get("whitelist", {}).get("qa", {}) or {}).items():
            if not isinstance(val, dict):
                continue
            cat = str(val.get("category", "") or "").lower().strip()
            if not cat:
                continue
            n = int(val.get("count", 0) or 0)
            if n <= 0:
                continue
            totals[cat] += n
        rate = {}
        for cat, total in totals.items():
            if total < min_n:
                continue
            rate[cat] = fails.get(cat, 0) / total
        return rate

    def is_bad_qa(self, question):
        norm = question.lower().strip()
        for t in getattr(self, "_qa_whitelist_flat", []):
            if t in norm or norm in t:
                return False
        return norm in self._qa_block

    def is_summary_risky(self):
        return getattr(self, "_sum_risky", False)

    def is_whitelisted(self, task, sig):
        return sig in self._wl_cache.get(task, set())

    # ---- analytics ---------------------------------------------------

    def get_analytics(self):
        out = {}
        for task, an in self.data.get("analytics", {}).items():
            total = an.get("total_fails", 0) + an.get("total_successes", 0)
            out[task] = {
                "total_observed": total,
                "success_rate": round(an.get("total_successes", 0) / max(total, 1), 3),
                "hard_fail_rate": round(an.get("hard_fails", 0) / max(total, 1), 3),
                "downgrade_rate": round(an.get("downgrades", 0) / max(total, 1), 3),
                "block_patterns": len([p for p in self.data.get("tasks", {}).get(task, {}).get("patterns", [])
                                       if p.get("severity", 0) > self.BLOCK_THRESHOLDS.get(
                                           task, {"min_severity": 0.65})["min_severity"]]),
                "whitelist_entries": len(self.data.get("whitelist", {}).get(task, {})),
            }
        return out

    def get_event_counts(self):
        if not hasattr(self, "_event_counters"):
            return {
                "ner_block": 0, "ner_success": 0, "ner_downgrade": 0,
                "re_block": 0, "re_success": 0, "re_downgrade": 0,
                "qa_block": 0, "qa_success": 0,
                "summary_fail": 0, "cross_task_propagations": 0,
            }
        return dict(self._event_counters)

    # ---- compatibility shims ----------------------------------------

    def set_step(self, step_num):
        if step_num % 100 == 0:
            self.save()

    def decay_by_step(self):
        pass  # replaced by real-time decay in _prune

    def is_flagged(self, flag_type, risk_id):
        return False


if __name__ == "__main__":
    # Minimal smoke test / usage example.
    import tempfile, os as _os
    tmp = tempfile.mkdtemp()
    bpm = BadPatternMemory(path=_os.path.join(tmp, "bpm_demo.json"))

    # Record a success and a repeated failure for NER.
    bpm.record("ner", {"pattern": "covid-19", "category": "Diagnosis"}, failure_type="success")
    for _ in range(4):
        bpm.record("ner", {"pattern": "patient", "category": "Diagnosis",
                           "evidence_source": "verifier_mmed"}, failure_type="hard_fail")
    bpm.save()

    print("gate_ner('covid-19') :", bpm.gate_ner("covid-19", "Diagnosis"))   # ALLOW (whitelisted)
    print("gate_ner('patient')  :", bpm.gate_ner("patient", "Diagnosis"))    # BLOCK/DOWNGRADE
    print("m3 violation TREATS Diagnosis->Symptom:",
          m3_ontology_violation("Symptom", "TREATS", "Diagnosis"))           # True (illegal)
    print("m1 error class:", m1_assign_error_class("Q?", "fever and cough", "patient denied fever"))
    print("analytics:", json.dumps(bpm.get_analytics(), indent=2))
