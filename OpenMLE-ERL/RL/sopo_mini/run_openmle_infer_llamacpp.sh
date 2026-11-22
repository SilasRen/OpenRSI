#!/bin/bash
# Same experiment as run_openmle_infer.sh, but the policy is served by llama.cpp (llama-server) from the
# released GGUF, one server per GPU, one seed per server. Use this when Apptainer/vLLM cannot run.
#
#   sbatch -p <gpu partition> --gres=gpu:tesla:2 --cpus-per-task=32 --time=48:00:00 run_openmle_infer_llamacpp.sh
set -euo pipefail
W=${W:-/online1/hitzy02/hitzy02}
PY=${PY:-$W/envs/sopo/bin/python}
PROG_PY=${PROG_PY:-$W/envs/sandbox/bin/python}                               # python that runs the generated programs (sandbox packages)
BUILD_ENV=${BUILD_ENV:-$W/envs/build}                 # conda env that compiled llama.cpp (runtime libs)
SERVER=${SERVER:-$W/llama.cpp/build/bin/llama-server}
GGUF=${GGUF:-$W/models/Frontis-MA1-30B-GGUF/Frontis-MA1-30B-Q4_K_M.gguf}
TASKS=${TASKS:-$W/openmle_tasks}
OUT=${OUT:-$W/infer_openmle}
SEEDS=${SEEDS:-"0 1"}                                 # one llama-server + one run per seed, GPU i for seed i
SLOTS=${SLOTS:-12}                                    # parallel sequences per server
SLOT_CTX=${SLOT_CTX:-32768}
EXTRA=${EXTRA:-}

# Like the original sandbox: shared HF/torch caches and model downloads through hf-mirror (SANDBOX_ONLINE=0 to disable).
export SANDBOX_ONLINE=${SANDBOX_ONLINE:-1} SANDBOX_PROXY=${SANDBOX_PROXY:-}
export SANDBOX_HF_HOME=${SANDBOX_HF_HOME:-$W/sandbox_cache/hf_home} SANDBOX_TORCH_HOME=${SANDBOX_TORCH_HOME:-$W/sandbox_cache/torch_home}
mkdir -p "$SANDBOX_HF_HOME" "$SANDBOX_TORCH_HOME"
export no_proxy=127.0.0.1,localhost${no_proxy:+,$no_proxy} NO_PROXY=127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}  # local server must bypass the cluster proxy
cd "${SOPO_DIR:-$W/sopo_rl/sopo_mini}"
mkdir -p "$OUT"
PIDS=()
cleanup() { for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

i=0
for s in $SEEDS; do
  port=$((8000 + i))
  # build-env libs only for the server: exported globally they break system curl and python
  LD_LIBRARY_PATH=$BUILD_ENV/lib:${LD_LIBRARY_PATH:-} CUDA_VISIBLE_DEVICES=$i "$SERVER" -m "$GGUF" --alias Frontis-MA1-30B -ngl 999 -fa on --jinja \
    -np "$SLOTS" -c $((SLOTS * SLOT_CTX)) --host 127.0.0.1 --port "$port" > "$OUT/llama_server_s$s.log" 2>&1 &
  PIDS+=($!)
  i=$((i + 1))
done
i=0
for s in $SEEDS; do
  port=$((8000 + i))
  for t in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$port/health" >/dev/null && break
    sleep 10
  done
  curl -sf "http://127.0.0.1:$port/health" >/dev/null || { echo "llama-server $s did not start"; tail -30 "$OUT/llama_server_s$s.log"; exit 1; }
  i=$((i + 1))
done
echo "llama-server ready"

i=0
RUNS=()
for s in $SEEDS; do
  port=$((8000 + i))
  JOBTMP=${TMPDIR:-/tmp}/openmle_jobs_${SLURM_JOB_ID:-local}_s$s
  mkdir -p "$JOBTMP"
  PROGRAM_THREADS=${PROGRAM_THREADS:-4} $PY infer_credit.py \
    --openmle-tasks "$TASKS" --out "${OUT}_s$s" --seed "$s" \
    --model-url "http://127.0.0.1:$port/v1" --model-name Frontis-MA1-30B \
    --program-python "$PROG_PY" --job-workdir "$JOBTMP" \
    --improve-only --parent-range ${PARENT_RANGE:-0.0,1.0} --num-states ${NUM_STATES:-16} --group-size ${GROUP:-8} \
    --steps 2 --repeats ${REPEATS:-2} --seed-drafts ${SEED_DRAFTS:-8} \
    --max-new-tokens ${MAX_NEW:-24576} --temperature 1.0 --timeout ${JOB_TIMEOUT:-3600} \
    --workers ${WORKERS:-4} --concurrency "$SLOTS" $EXTRA > "${OUT}_s$s.log" 2>&1 &
  RUNS+=($!)
  i=$((i + 1))
done
status=0
for p in "${RUNS[@]}"; do wait "$p" || status=1; done
exit $status
