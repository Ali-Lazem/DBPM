"""
example_usage.py — minimal end-to-end demonstration of DBPM gating.

Shows the fail/success learning for all task channels and the three-tier
gate (BLOCK / DOWNGRADE / ALLOW). Run: python3 example_usage.py
"""
from dbpm import BadPatternMemory, m3_ontology_violation, m1_assign_error_class
import tempfile, os

tmp = tempfile.mkdtemp()
bpm = BadPatternMemory(path=os.path.join(tmp, "bpm_example.json"))

# 1) NER: one success (whitelist) + repeated failures (blocklist)
bpm.record("ner", {"pattern": "covid-19", "category": "Diagnosis"}, failure_type="success")
for _ in range(4):
    bpm.record("ner", {"pattern": "patient", "category": "Diagnosis",
                       "evidence_source": "verifier_mmed"}, failure_type="hard_fail")

# 2) RE: record an ontology-violating relation as a hard fail
for _ in range(3):
    bpm.record("re", {"head_type": "Symptom", "relation": "TREATS", "tail_type": "Diagnosis",
                      "head": "fever", "tail": "covid-19", "onto_violation": True,
                      "evidence_source": "rule_based"}, failure_type="hard_fail")

# 3) QA: a rejected answer carries an error_class; a good one is a success
bpm.record("qa", {"question": "What treatments?", "answer": "aspirin",
                  "category": "treatment", "error_class": "unsupported"}, failure_type="hard_fail")
bpm.record("qa", {"question": "What diagnoses?", "answer": "ARDS",
                  "category": "diagnosis"}, failure_type="success")

# 4) summary / medications / temporal_events all use the same API
bpm.record("summary", {"source_len": 1200}, failure_type="success")
bpm.record("medications", {"drug": "vancomycin"}, failure_type="success")
bpm.record("temporal_events", {"event_type": "ADMISSION"}, failure_type="success")

bpm.save()

print("NER gate covid-19 :", bpm.gate_ner("covid-19", "Diagnosis"))   # ALLOW
print("NER gate patient  :", bpm.gate_ner("patient", "Diagnosis"))    # BLOCK
print("RE  gate S-TREATS-D:", bpm.gate_relation("Symptom", "TREATS", "Diagnosis"))
print("QA  gate treatment :", bpm.gate_qa("treatment", difficulty=0.5, ner_categories={"diagnosis"}))
print()
print("predicate m3 (illegal):", m3_ontology_violation("Symptom", "TREATS", "Diagnosis"))
print("predicate m1:", m1_assign_error_class("Q?", "no fever", "patient had fever"))
print()
import json
print("event counts:", json.dumps(bpm.get_event_counts(), indent=2))
