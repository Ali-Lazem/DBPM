#!/bin/bash
#SBATCH --job-name=DBPM_FULL_ON_vs_OFF_5K
#SBATCH --partition=<GPU_PARTITION>          # e.g. a partition with 4x H200 + GPUs
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --output=dbpm_ablation_%j.out
#
# DBPM full-ON vs full-OFF ablation on a 5,000-patient cohort.
#
# This script reproduces the paired ablation behind the paper's headline
# selectivity number. It starts two vLLM servers (a generator and a verifier),
# then runs the extraction pipeline twice on the same cohort: once with all
# DBPM mechanisms ENABLED (RUN A) and once with them all DISABLED (RUN B).
#
# NOTE: the extraction pipeline/worker is NOT included in this repository (see
# the data-availability statement). The placeholders below mark where your own
# pipeline entry point and model paths go. The DBPM gating module this ablation
# toggles is provided as `dbpm.py`.

set -euo pipefail

# ── ENVIRONMENT ───────────────────────────────────────────────────────
module purge
module load <PYTHON_MODULE> <CUDA_MODULE>
source <PATH_TO_VENV>/bin/activate
cd <PATH_TO_PIPELINE_CODE>

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

GENERATOR_MODEL=<PATH_TO_GENERATOR_MODEL>     # e.g. Llama-3.3-70B
VERIFIER_MODEL=<PATH_TO_VERIFIER_MODEL>       # e.g. MMed-Llama-3.1-70B
UID_LIST=<PATH_TO>/ablation_5k_uids.json      # cohort UID list (see cohorts/)
PIPELINE_ENTRY=<your_pipeline_entry>.py       # NOT included in this repo

cleanup() {
    [ -n "${WORKER_PID:-}" ]   && kill $WORKER_PID   2>/dev/null || true
    [ -n "${VERIFIER_PID:-}" ] && kill $VERIFIER_PID 2>/dev/null || true
    sleep 2
    [ -n "${WORKER_PID:-}" ]   && kill -9 $WORKER_PID   2>/dev/null || true
    [ -n "${VERIFIER_PID:-}" ] && kill -9 $VERIFIER_PID 2>/dev/null || true
}
trap cleanup EXIT

# ── START vLLM SERVERS (TP=2 each: generator on 0,1; verifier on 2,3) ──
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model "$GENERATOR_MODEL" --port 8000 --host 0.0.0.0 \
    --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 --dtype bfloat16 --enable-chunked-prefill \
    > generator_vllm.log 2>&1 &
WORKER_PID=$!

CUDA_VISIBLE_DEVICES=2,3 python -m vllm.entrypoints.openai.api_server \
    --model "$VERIFIER_MODEL" --port 8001 --host 0.0.0.0 \
    --tensor-parallel-size 2 --max-model-len 8192 --max-num-seqs 256 \
    --gpu-memory-utilization 0.90 --dtype bfloat16 --enable-chunked-prefill \
    > verifier_vllm.log 2>&1 &
VERIFIER_PID=$!

wait_for_server() {
    local PORT=$1 NAME=$2 PID=$3 LOG=$4
    for i in $(seq 1 120); do
        curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1 && { echo "$NAME ready"; return 0; }
        kill -0 $PID 2>/dev/null || { echo "$NAME crashed"; tail -30 "$LOG"; exit 1; }
        sleep 5
    done
    echo "Timeout waiting for $NAME"; exit 1
}
wait_for_server 8000 "Generator" $WORKER_PID "generator_vllm.log"
wait_for_server 8001 "Verifier"  $VERIFIER_PID "verifier_vllm.log"

# ── GUARD: test cohort disjoint from dev splits ───────────────────────
python3 - "$UID_LIST" <<'PY'
import json, sys
test = set(json.load(open(sys.argv[1]))["uids"])
for devf in ("dev_2k_uids.json", "dev_mini_200_uids.json"):
    try:
        dev = set(json.load(open(devf))["uids"])
    except FileNotFoundError:
        continue
    ov = dev & test
    assert not ov, f"ABORT: {devf} overlaps test by {len(ov)}"
print("GUARD ok: test cohort disjoint from dev splits")
PY

# ── SHARED ENV ────────────────────────────────────────────────────────
export ABLATION_UID_LIST="$UID_LIST"
export TARGET_N=5000

# ── RUN A: DBPM FULL ON (all mechanisms active) ───────────────────────
echo "=== RUN A: DBPM FULL ON ==="
export OUTPUT_DIR_NAME="risk_DBPM_full_ON"
export BPM_ENABLE_CROSS_TASK=1
export BPM_ENABLE_UNCERTAINTY=1
export BPM_ENABLE_THREE_TIER=1
export BPM_ENABLE_SOURCE_WEIGHTS=1
unset  BPM_DISABLE
unset  BPM_DISABLE_M3
export M1V7_ENABLE=1
export M1V6_ENABLE=1
timeout --kill-after=120s -s SIGTERM 11h python "$PIPELINE_ENTRY" > run_full_ON.log 2>&1
echo "RUN A exit: $?"

# ── RUN B: DBPM FULL OFF (master switch + M3 + M1 gates disabled) ─────
echo "=== RUN B: DBPM FULL OFF ==="
export OUTPUT_DIR_NAME="risk_DBPM_full_OFF"
export BPM_DISABLE=1            # flips cross_task, uncertainty, three_tier, source_weights OFF
export BPM_DISABLE_M3=1
export M1V7_ENABLE=0
export M1V6_ENABLE=0
timeout --kill-after=120s -s SIGTERM 11h python "$PIPELINE_ENTRY" > run_full_OFF.log 2>&1
echo "RUN B exit: $?"

echo "Done. Grade with graders/aggregate_dbpm_provenance.py and graders/grade_m1v7_selectivity.py"
