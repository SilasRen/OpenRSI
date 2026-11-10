#!/usr/bin/env bash
# One-node, offline SOPO validation: no Docker, no sandbox service, no external API.
#
#   sbatch --gres=gpu:8 --cpus-per-task=64 --mem=400G --time=24:00:00 run_experiment.sh
#   # or directly on an allocated node:  bash run_experiment.sh
#
# Arms (one GPU each, SEEDS seeds per arm; default 4 arms x 2 seeds = 8 GPUs):
#   control        original single-step reward (SOPO_STEPS=0)
#   control_3x     same reward, 3x the RL steps = same number of program executions as SOPO
#   sopo           SOPO simple version: reward = best utility the branch reaches in 2 more steps
#   sopo_cur       sopo + curriculum on credit variance
# Then every final checkpoint and the untrained base model run the same fixed-budget
# search on held-out tasks, and report.py writes summary.md with paired 95% CIs.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"   # local path recommended on offline clusters
OUT="${OUT:-$PWD/sopo_mini_runs}"
PY="${PY:-python}"
NGPU="${NGPU:-$( (nvidia-smi -L 2>/dev/null || true) | wc -l)}"
SEEDS="${SEEDS:-2}"
STEPS="${STEPS:-60}"
N_GROUPS="${N_GROUPS:-4}"
GROUP_SIZE="${GROUP_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
TIMEOUT="${TIMEOUT:-90}"
LR="${LR:-1e-6}"
EVAL_BUDGET="${EVAL_BUDGET:-16}"
EVAL_SEEDS="${EVAL_SEEDS:-3}"
ARMS="${ARMS:-control control_3x sopo sopo_cur}"
CPUS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

if [ "${QUICK:-0}" = "1" ]; then          # ~1-2 h smoke run to check the cluster setup
  MODEL="${MODEL_QUICK:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"; STEPS=8; SEEDS=1; EVAL_SEEDS=1; EVAL_BUDGET=8
  TRAIN_PER_KIND=3; EVAL_PER_KIND=2
fi
TRAIN_PER_KIND="${TRAIN_PER_KIND:-10}"
EVAL_PER_KIND="${EVAL_PER_KIND:-4}"

EXTRA_TRAIN=""
if [ "${FAKE:-0}" = "1" ]; then           # pipeline test without GPU or model weights
  MODEL=fake; NGPU="${NGPU_FAKE:-2}"; EXTRA_TRAIN="--fake-policy"
fi
[ "$NGPU" -ge 1 ] || { echo "no GPU visible" >&2; exit 1; }
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/logs" "$OUT/eval"
[ "${FAKE:-0}" = "1" ] || "$PY" -c "import torch, transformers, sklearn, pandas; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [ ! -f "$OUT/tasks/splits.json" ]; then
  "$PY" "$HERE/tasks.py" --out "$OUT/tasks" --train-per-kind "$TRAIN_PER_KIND" --eval-per-kind "$EVAL_PER_KIND"
fi

arm_args() {
  case "$1" in
    control)    echo "--sopo-steps 0 --steps $STEPS" ;;
    control_3x) echo "--sopo-steps 0 --steps $((STEPS * 3))" ;;
    sopo)       echo "--sopo-steps 2 --reward-mode best --steps $STEPS" ;;
    sopo_cur)   echo "--sopo-steps 2 --reward-mode best --curriculum --steps $STEPS" ;;
    sopo_credit) echo "--sopo-steps 2 --reward-mode credit --lam 0.9 --steps $STEPS" ;;
    *) echo "unknown arm $1" >&2; exit 2 ;;
  esac
}

# ---------------------------------------------------------------- training
jobs_list=()
for arm in $ARMS; do for s in $(seq 0 $((SEEDS - 1))); do jobs_list+=("$arm:$s"); done; done
WORKERS=$(( CPUS / (NGPU < ${#jobs_list[@]} ? NGPU : ${#jobs_list[@]}) ))
[ "$WORKERS" -ge 1 ] || WORKERS=1

run_wave() {  # run "cmd" entries in parallel, at most NGPU at a time, one GPU each
  local -n cmds=$1; local i=0 pids=()
  for c in "${cmds[@]}"; do
    gpu=$(( i % NGPU ))
    CUDA_VISIBLE_DEVICES=$gpu bash -c "$c" &
    pids+=($!); i=$((i + 1))
    if [ $(( i % NGPU )) -eq 0 ]; then for p in "${pids[@]}"; do wait "$p"; done; pids=(); fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

train_cmds=()
for js in "${jobs_list[@]}"; do
  arm="${js%%:*}"; s="${js##*:}"; dir="$OUT/train/${arm}_s$s"
  [ -f "$dir/final/config.json" ] || [ -f "$dir/final/FAKE" ] && continue
  train_cmds+=("$PY $HERE/train.py --tasks $OUT/tasks --out $dir --model $MODEL --seed $s \
    --groups $N_GROUPS --group-size $GROUP_SIZE --max-new-tokens $MAX_NEW_TOKENS --timeout $TIMEOUT \
    --workers $WORKERS --lr $LR $EXTRA_TRAIN $(arm_args "$arm") > $OUT/logs/train_${arm}_s$s.log 2>&1")
done
echo "training ${#train_cmds[@]} runs on $NGPU GPUs ($WORKERS CPU workers each)"
[ ${#train_cmds[@]} -eq 0 ] || run_wave train_cmds

# ---------------------------------------------------------------- evaluation
eval_cmds=()
add_eval() {  # name model
  [ -f "$OUT/eval/$1.jsonl" ] && return 0
  eval_cmds+=("$PY $HERE/evaluate.py --tasks $OUT/tasks --model $2 --out $OUT/eval/$1.jsonl \
    --budget $EVAL_BUDGET --seeds $EVAL_SEEDS --max-new-tokens $MAX_NEW_TOKENS --timeout $TIMEOUT \
    --workers $WORKERS > $OUT/logs/eval_$1.log 2>&1")
}
add_eval base "$MODEL"
for js in "${jobs_list[@]}"; do
  arm="${js%%:*}"; s="${js##*:}"; add_eval "${arm}_s$s" "$OUT/train/${arm}_s$s/final"
done
echo "evaluating ${#eval_cmds[@]} models"
[ ${#eval_cmds[@]} -eq 0 ] || run_wave eval_cmds

# Pool seeds of each arm into one file so the report compares arms, not single runs.
for arm in $ARMS; do
  : > "$OUT/eval/$arm.pooled"
  for s in $(seq 0 $((SEEDS - 1))); do
    "$PY" - "$OUT/eval/${arm}_s$s.jsonl" "$s" >> "$OUT/eval/$arm.pooled" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line); r["seed"] = f"train{sys.argv[2]}/{r['seed']}"; print(json.dumps(r))
EOF
  done
done
# Base model: replicate rows per training seed so pairing keys match the pooled arms.
: > "$OUT/eval/base.pooled"
for s in $(seq 0 $((SEEDS - 1))); do
  "$PY" - "$OUT/eval/base.jsonl" "$s" >> "$OUT/eval/base.pooled" <<'EOF'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line); r["seed"] = f"train{sys.argv[2]}/{r['seed']}"; print(json.dumps(r))
EOF
done
mkdir -p "$OUT/report"
for f in "$OUT"/eval/*.pooled; do cp "$f" "$OUT/report/$(basename "${f%.pooled}").jsonl"; done
"$PY" "$HERE/report.py" --baseline control --out "$OUT/report" \
  --budgets "1,4,$((EVAL_BUDGET / 2)),$EVAL_BUDGET" "$OUT"/report/*.jsonl
echo "done: $OUT/report/summary.md"
