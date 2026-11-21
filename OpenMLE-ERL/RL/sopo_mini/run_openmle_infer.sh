#!/bin/bash
# One Slurm job: serve Frontis-MA1 with vLLM on the job's GPUs, then run the inference test on real
# OpenMLE task packages with the paper's prompts, sandbox scoring and reward.
#
#   sbatch -p <gpu partition> --gres=gpu:tesla:2 --cpus-per-task=32 --time=48:00:00 run_openmle_infer.sh
#
# Settings come from the environment (defaults below). Nothing here downloads anything.
set -euo pipefail
W=${W:-/online1/hitzy02/hitzy02}
PY=${PY:-$W/envs/sopo/bin/python}            # env with httpx, black, pandas, scikit-learn, lightgbm, xgboost
SIF=${SIF:-$W/images/vllm.sif}               # apptainer pull $SIF docker://docker.m.daocloud.io/vllm/vllm-openai:v0.8.5.post1 (CUDA 12.4, driver 550)
VLLM=${VLLM:-"apptainer exec --nv --bind $W:$W $SIF vllm"}
MODEL=${MODEL:-$W/models/Frontis-MA1-30B}
TASKS=${TASKS:-$W/openmle_tasks}
OUT=${OUT:-$W/infer_openmle_s0}
SEED=${SEED:-0}
TP=${TP:-2}
PORT=${PORT:-8000}
MAX_LEN=${MAX_LEN:-40960}
EXTRA=${EXTRA:-}                              # extra infer_credit flags

module load amd/apptainer/1.2.5 2>/dev/null || true
cd "${SOPO_DIR:-$W/sopo_rl/sopo_mini}"   # sbatch runs a copy of this script, so do not use $0
mkdir -p "$OUT"
JOBTMP=${TMPDIR:-/tmp}/openmle_jobs_$SLURM_JOB_ID
mkdir -p "$JOBTMP"

$VLLM serve "$MODEL" --served-model-name Frontis-MA1-30B --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.90 --port "$PORT" \
  > "$OUT/vllm.log" 2>&1 &   # thinking text is stripped client-side (openmle_backend.strip_thinking)
VPID=$!
trap 'kill $VPID 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break
  kill -0 $VPID 2>/dev/null || { echo "vLLM exited, see $OUT/vllm.log"; tail -40 "$OUT/vllm.log"; exit 1; }
  sleep 10
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "vLLM did not start in 30 min"; exit 1; }
echo "vLLM ready"

PROGRAM_THREADS=${PROGRAM_THREADS:-6} $PY infer_credit.py \
  --openmle-tasks "$TASKS" --out "$OUT" --seed "$SEED" \
  --model-url "http://127.0.0.1:$PORT/v1" --model-name Frontis-MA1-30B \
  --program-python "$PY" --job-workdir "$JOBTMP" \
  --improve-only --parent-range 0.2,0.8 --num-states ${NUM_STATES:-16} --group-size ${GROUP:-8} \
  --steps 2 --repeats ${REPEATS:-2} --seed-drafts ${SEED_DRAFTS:-8} \
  --max-new-tokens ${MAX_NEW:-24576} --temperature 1.0 --timeout ${JOB_TIMEOUT:-1200} \
  --workers ${WORKERS:-5} --concurrency ${CONCURRENCY:-48} $EXTRA
