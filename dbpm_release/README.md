# DBPM — Dynamic Bidirectional Pattern Memory

Reference implementation of the DBPM gating module from
**"[Paper 1 title]"** ([authors], [venue], 2026). Preprint: [arXiv DOI].

DBPM is an inference-time, gradient-free "cognitive memory" that gates the
outputs of a verifier-grounded clinical extraction pipeline. Per task it
records both **failures** (a blocklist) and **successes** (a whitelist),
propagates signal **across tasks**, weights evidence by **source**, and
**forgets** via real-time decay. The gate consults this memory to **BLOCK**,
**DOWNGRADE**, or **ALLOW** each candidate — no gradient updates.

## What this is
The full DBPM gating logic for all seven task channels (NER, RE, QA, summary,
medications, temporal_events, belief_update), including:

- Bidirectional fail/success learning
- Cross-task signal propagation (e.g. an RE hard-fail warns NER)
- Three-tier gating (BLOCK / DOWNGRADE / ALLOW)
- Bayesian source weighting
- Real-time exponential decay (UTC half-life)
- The two verifier-independent predicates used by the QA gate (M1) and the
  RE ontology channel (M3)

## What this is NOT
The gating module only. The surrounding extraction pipeline (orchestration,
model serving, the multi-stage extraction loop, the risk/intelligence layer,
and the visualisation builder) is a separate system and is not included.

## Install & run
```bash
pip install -r requirements.txt   # just `filelock`
python3 example_usage.py          # end-to-end demo across all task channels
python3 dbpm.py                   # built-in smoke test
```

## Quick start
```python
from dbpm import BadPatternMemory

bpm = BadPatternMemory(path="bpm.json")

# learn from outcomes
bpm.record("ner", {"pattern": "covid-19", "category": "Diagnosis"}, failure_type="success")
bpm.record("ner", {"pattern": "patient",  "category": "Diagnosis",
                   "evidence_source": "verifier_mmed"}, failure_type="hard_fail")
bpm.save()

# gate future candidates
bpm.gate_ner("covid-19", "Diagnosis")          # -> "ALLOW"
bpm.gate_relation("Symptom", "TREATS", "Diagnosis")  # -> "DOWNGRADE" / "BLOCK"
bpm.gate_qa("treatment", difficulty=0.5, ner_categories={"diagnosis"})
```

## Mapping to the paper
| Paper element | Code |
|---|---|
| Algorithm 1 (memory update) | `BadPatternMemory.record` |
| Three-tier gate | `gate_ner` / `gate_relation` / `gate_qa` |
| Bayesian source weighting | `SOURCE_WEIGHTS` + `record` |
| Real-time decay (half-life) | `_time_decay` |
| Cross-task propagation | `record` (cross-task path) |
| M1 QA error predicate | `m1_classify_qa_error` |
| M3 RE ontology predicate | `m3_ontology_violation` |

## Ablation flags
All default ON; set to `0` to disable. `BPM_DISABLE=1` is the master switch.
`BPM_ENABLE_CROSS_TASK`, `BPM_ENABLE_UNCERTAINTY`, `BPM_ENABLE_THREE_TIER`,
`BPM_ENABLE_SOURCE_WEIGHTS`, `BPM_DISABLE_M3`.

Note: the `verifier_self` entry in `SOURCE_WEIGHTS` is **reserved for the
optional self-verification ablation arm and is disabled in the reported
configuration**.

## Reproducing the reported gating numbers
The aggregate gating-outcome counts that reproduce the paper's lift,
confidence interval, and per-category breakdown are in
`results/m1v7_full_ON_selectivity.json` (counts and rates only; no clinical
text). The cohort is indexed by the UID lists in `cohorts/`, which point into
the public PMC-Patients corpus.

## Hardware requirements

**This module (`dbpm.py`) and the grader scripts:** none special — they run
on any machine with Python 3.8+. No GPU required.

**The extraction pipeline that produced the analysed data** (not included in
this repository) was run on 4× NVIDIA H200 GPUs (141 GB each): a Llama-3.3-70B
generator and an MMed-Llama-3.1-70B verifier, each tensor-parallel across two
GPUs, with ~64 CPU cores and ~900 GB RAM. See `run_ablation.sh` for the
serving configuration.

## License
See [LICENSE](LICENSE).

## Ablation protocol
`run_ablation.sh` reproduces the full-ON vs full-OFF paired ablation: it starts
the two vLLM servers and runs the pipeline twice on the same cohort, toggling
the DBPM mechanisms via the `BPM_*` environment flags. The extraction pipeline
entry point and model paths are placeholders (`<...>`) because the pipeline
itself is not included in this repository; the DBPM gating module it toggles is
`dbpm.py`. Grade the two runs with the scripts in `graders/`.

## Repository layout
```
dbpm.py                 standalone DBPM gating module (the contribution)
example_usage.py        end-to-end demo across all task channels
run_ablation.sh         full-ON vs full-OFF ablation protocol (genericized)
graders/                scripts that reproduce the reported numbers from outputs
cohorts/                cohort UID lists (index the public PMC-Patients corpus)
results/                aggregate gating-outcome counts (counts/rates only)
config/                 (optional) flags and run configuration
CITATION.cff            citation metadata
```
