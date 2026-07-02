<!-- Badges: place at very top, immediately under the H1 title in the rendered README.
     Replace ZENODO_ID with the real DBPM Zenodo deposit id once minted. -->

# DBPM — Dynamic Bidirectional Pattern Memory

### Verifier-grounded inference-time gating for clinical extraction pipelines
[![arXiv](https://img.shields.io/badge/arXiv-2607.00870-b31b1b.svg)](https://arxiv.org/abs/2607.00870)
[![DOI](https://zenodo.org/badge/1262243839.svg)](https://doi.org/10.5281/zenodo.21038023)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![Paper](https://img.shields.io/badge/paper-in%20preparation-lightgrey.svg)](#citation)

Reference implementation of the DBPM gating module from *"Dynamic Bidirectional
Pattern Memory: A Production-Scale Empirical Characterisation of Inference-Time
Gating in Clinical NLP"* (Lazem & Teahan, 2026). Preprint: *arXiv (in preparation)*.

DBPM is an inference-time, gradient-free pattern memory that gates the outputs of a
verifier-grounded clinical extraction pipeline. Per task it records both **failures**
(a blocklist) and **successes** (a whitelist), propagates signal **across tasks**,
weights evidence by **source reliability**, and **forgets** via real-time decay. The
gate consults this memory to **BLOCK**, **DOWNGRADE**, or **ALLOW** each candidate,
with no gradient updates to either the generator or the verifier.

---

## The finding in one table

The pipeline pairs a generator (**Llama-3.3-70B**) with a verifier
(**MMed-Llama-3.1-70B**) over the **167,034** patient narratives of PMC-Patients. The
paper characterises which gating signals actually make such a memory selective at
production scale. Two results anchor it:

| Finding | Result |
| --- | --- |
| **Natural verifier-fed design fails at scale** | The relation-extraction channel persists **0** patterns despite **785,797** logged rejections, a structural consequence of decay outrunning reinforcement on a diffuse signal. |
| **One signal source is selective** | Of five tested question-answering gate signals, only the NER-coverage signal separates pass from fail: path lift **1.84** (95% CI [1.75, 1.93], *p* < 10⁻¹⁵), within a **1.52–1.84** band across four replications. |

The organising principle: a pre-generation gate is selective for verifier rejection
**only when its signal probes the same grounding question the verifier itself
evaluates**. A verifier-independent ontology predicate fills the empty relation
channel where the verifier-fed signal could not.

---

## What this is

The standalone DBPM gating module. The deployed worker gates **three** task channels
at the candidate level, **NER, RE, and QAR**; the remaining task buckets carry
reserved threshold entries but no consumer gate (see *Mapping to the paper*). The
module implements:

- Bidirectional fail/success learning (blocklist and whitelist)
- Cross-task signal propagation (e.g. an RE hard-fail nudges NER severity)
- Three-tier gating (BLOCK / DOWNGRADE / ALLOW)
- Source-reliability weighting of evidence
- Real-time exponential decay (wall-clock half-life)
- The two verifier-independent predicates: the NER-coverage signal for the QA gate
  and the ontology-violation predicate for the RE channel

> **Note on naming.** The paper refers to the mechanism as DBPM throughout; the code
> instantiates the class `BadPatternMemory` and the QA gate method is `gate_qa`, both
> retained as legacy identifiers. The paper's Appendix A.1 gives the full
> concept-to-identifier mapping.

## What this is NOT

The gating module only. The surrounding extraction pipeline (orchestration, model
serving, the multi-stage extraction loop, the risk/intelligence layer, and the
visualisation builder) is a separate system and is **not** included here. The cohort
UID lists and aggregate counts permit reproduction of the reported gating numbers
without the pipeline.

---

## How DBPM sits in the pipeline

```
                  candidates                      verdicts
   Generator ───────────────▶  DBPM gate  ──────────────▶  Verifier
   (Llama-3.3-70B)             │   │   │                   (MMed-Llama-3.1-70B)
                               │   │   └─ ALLOW   ─▶ candidate passes unchanged
                               │   └──── DOWNGRADE ─▶ candidate passes, tagged
                               └──────── BLOCK     ─▶ candidate dropped (verifier not called)
                                   ▲
                                   │  consults
                          ┌────────┴─────────┐
                          │  Blocklist  B_j  │  severity per signature
                          │  Whitelist  W_j  │  confidence per signature
                          └──────────────────┘
                                   ▲
                          verdict feedback (no gradient updates)
```

---

## Repository layout

| Path | Contents |
| --- | --- |
| **`dbpm.py`** | Standalone DBPM gating module (the contribution); all numeric constants as class attributes |
| **`example_usage.py`** | End-to-end demo across the gated task channels |
| **`run_ablation.sh`** | Full-ON vs full-OFF ablation protocol (genericised; pipeline entry points are placeholders) |
| **`graders/`** | Scripts that reproduce the reported lift, CI, and per-category numbers from released outputs |
| **`cohorts/`** | Held-out and development cohort UID lists (index the public PMC-Patients corpus) |
| **`results/`** | Aggregate gating-outcome counts (counts and rates only; no clinical text) |
| **`requirements.txt`** | Dependencies (`filelock`; `numpy` for one grader) |
| **`CITATION.cff`** | Citation metadata |

---

## Install & run

```bash
pip install -r requirements.txt   # filelock (core); numpy (one grader)
python3 example_usage.py          # end-to-end demo across the gated channels
python3 dbpm.py                   # built-in smoke test
```

No GPU required for the module or the graders; they run on any machine with Python 3.8+.

## Quick start

```python
from dbpm import BadPatternMemory   # the DBPM class (legacy name)

bpm = BadPatternMemory(path="bpm.json")

# learn from verifier outcomes
bpm.record("ner", {"pattern": "covid-19", "category": "Diagnosis"},
           failure_type="success")
bpm.record("ner", {"pattern": "patient", "category": "Diagnosis",
                   "evidence_source": "verifier_mmed"}, failure_type="hard_fail")
bpm.save()

# gate future candidates
bpm.gate_ner("covid-19", "Diagnosis")                # -> "ALLOW"
bpm.gate_relation("Symptom", "TREATS", "Diagnosis")  # -> "DOWNGRADE" / "BLOCK"
bpm.gate_qa("treatment", difficulty=0.5, ner_categories={"diagnosis"})
```

---

## Mapping to the paper

| Paper element | Code |
| --- | --- |
| Algorithm 1 (memory update operator `U`) | `BadPatternMemory.record` |
| Three-tier gate (Eq. 4) | `gate_ner` / `gate_relation` / `gate_qa` |
| Source-reliability weights (Table 3) | `SOURCE_WEIGHTS` + `record` |
| Real-time decay, wall-clock half-life (Eq. 5) | `_time_decay` |
| Cross-task propagation (§4.4) | `record` (cross-task path) |
| NER-coverage QA signal (the working gate, §4.8) | `gate_qa` coverage branch |
| Ontology-violation RE predicate (§4.7) | `m3_ontology_violation` |

The three gated tasks are NER, RE, and QAR. The `summary` channel has a reserved
threshold but no consumer gate; medication, temporal-event, risk, and visualisation
stages are outside the gating system (paper Appendix A.6).

---

## Ablation flags

All default ON; set any to `0` to disable. `BPM_DISABLE=1` is the master switch.

| Flag | Controls |
| --- | --- |
| `BPM_ENABLE_CROSS_TASK` | cross-task signal propagation |
| `BPM_ENABLE_UNCERTAINTY` | uncertainty-weighted updates |
| `BPM_ENABLE_THREE_TIER` | the DOWNGRADE tier (off = BLOCK/ALLOW only) |
| `BPM_ENABLE_SOURCE_WEIGHTS` | source-reliability weighting |
| `BPM_DISABLE_M3` | the RE ontology-violation channel |
| `M1_QA_GATE_HARD_BLOCK` | whether the QA gate may emit BLOCK (default `0`, DOWNGRADE-first) |
| `BPM_DISABLE` | master switch: disables all gating and recording |

> The `verifier_self` entry in `SOURCE_WEIGHTS` is **reserved for the optional
> self-verification ablation arm and is disabled in the reported configuration**.

---

## Reproducing each reported number

| Result | Command |
| --- | --- |
| Five-signal selectivity dissection | `python3 graders/grade_m1v7_selectivity.py --json results/m1v7_full_ON_selectivity.json` |
| Path-lift 95% confidence interval | `python3 graders/compute_v7_lift_ci.py --json results/m1v7_full_ON_selectivity.json` |
| 167K verifier-fed channel cross-check | `python3 graders/crosscheck_F1_167k_blocklist.py` |
| Full-ON vs full-OFF ablation | `bash run_ablation.sh` (pipeline placeholders; toggles `dbpm.py`) |

The aggregate gating-outcome counts behind the lift, confidence interval, and
per-category breakdown are in `results/m1v7_full_ON_selectivity.json` (counts and
rates only; no clinical text). The cohort is indexed by the UID lists in `cohorts/`,
which point into the public PMC-Patients corpus.

---

## Ablation protocol

`run_ablation.sh` reproduces the full-ON vs full-OFF paired ablation: it starts the
two vLLM servers and runs the pipeline twice on the same cohort, toggling the DBPM
mechanisms via the `BPM_*` environment flags. The pipeline entry point and model
paths are placeholders (`<...>`) because the pipeline itself is not included; the
DBPM gating module it toggles is `dbpm.py`. Grade the two runs with the scripts in
`graders/`.

---

## Hardware requirements

**This module (`dbpm.py`) and the graders:** none special. Python 3.8+, no GPU.

**The extraction pipeline that produced the analysed data** (not included here) ran
on 4× NVIDIA H200 GPUs (141 GB HBM3e each): a Llama-3.3-70B generator and an
MMed-Llama-3.1-70B verifier, each tensor-parallel across two GPUs, with ~64 CPU cores
and ~900 GB RAM. The full run processed 167,034 patients over approximately twelve
days of elapsed wall-clock time. See `run_ablation.sh` for the serving configuration.

---

## Citation

```bibtex
@article{lazem2026dbpm,
  title         = {Dynamic Bidirectional Pattern Memory: A Production-Scale
                   Empirical Characterisation of Inference-Time Gating in Clinical NLP},
  author        = {Lazem, Ali H. and Teahan, William J.},
  journal       = {arXiv preprint arXiv:2607.00870},
  year          = {2026},
  eprint        = {2607.00870},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}```

See [`CITATION.cff`](CITATION.cff) for machine-readable metadata.

## License

Code is licensed under [MIT](LICENSE). The accompanying paper is licensed under CC BY 4.0.

## Acknowledgements

Computation performed on **Supercomputing Wales** (Falcon, project SCWF00175), with
support from the **Bangor eResearch Team**, including Dr Ade Fewings.
