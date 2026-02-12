#!/usr/bin/env bash
set -euo pipefail

WANDB_PROJECT="Maze-PTRM-Ablations"
RUN_PREFIX="spec"

SOURCE_REPO="sapientinc/maze-30x30-hard-1k"
DATA_DIR="data/maze-30x30-hard-1k"
SKIP_DATASET=0
AUG=true
SUBSAMPLE_SIZE=""

EPOCHS=2000
EVAL_INTERVAL=200
GLOBAL_BATCH_SIZE=128

LR=1e-4
PUZZLE_EMB_LR=1e-4
WEIGHT_DECAY=1.0
PUZZLE_EMB_WEIGHT_DECAY=1.0

L_LAYERS=2
H_CYCLES=3
L_CYCLES=4
EMA=true

K_SWEEP="8 16 32 64"

PYTHON_BIN="${PYTHON_BIN:-}"
TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
NPROC_PER_NODE=1

usage() {
  cat <<'EOF'
Usage: bash scripts/run_spec_ptrm_ablations.sh [options]

Options:
  --wandb-project <name>       W&B project name (default: Maze-PTRM-Ablations)
  --run-prefix <name>          Run name prefix (default: spec)
  --source-repo <repo>         HF dataset repo (default: sapientinc/maze-30x30-hard-1k)
  --data-dir <path>            Dataset output path (default: data/maze-30x30-hard-1k)
  --skip-dataset               Skip dataset build/download
  --aug <true|false>           Enable train dihedral augmentation (default: true)
  --subsample-size <int>       Optional train subsample for dataset build

  --epochs <int>               Training epochs per run (default: 2000)
  --eval-interval <int>        Eval interval (default: 200)
  --global-batch-size <int>    Global batch size (default: 128)
  --lr <float>                 Model lr (default: 1e-4)
  --puzzle-emb-lr <float>      Puzzle embedding lr (default: 1e-4)
  --weight-decay <float>       Model weight decay (default: 1.0)
  --puzzle-emb-weight-decay <float> Puzzle embedding weight decay (default: 1.0)
  --l-layers <int>             arch.L_layers (default: 2)
  --h-cycles <int>             arch.H_cycles (default: 3)
  --l-cycles <int>             arch.L_cycles (default: 4)
  --ema <true|false>           EMA toggle (default: true)

  --k-sweep "<vals>"           PTRM z-slot values (default: "8 16 32 64")
  --python-bin <path>          Python executable to use
  --torchrun-bin <name/path>   torchrun executable (default: torchrun)
  --nproc-per-node <int>       Use torchrun if >1 (default: 1)
  --help                       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --run-prefix) RUN_PREFIX="$2"; shift 2 ;;
    --source-repo) SOURCE_REPO="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --aug) AUG="$2"; shift 2 ;;
    --subsample-size) SUBSAMPLE_SIZE="$2"; shift 2 ;;

    --epochs) EPOCHS="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --puzzle-emb-lr) PUZZLE_EMB_LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --puzzle-emb-weight-decay) PUZZLE_EMB_WEIGHT_DECAY="$2"; shift 2 ;;
    --l-layers) L_LAYERS="$2"; shift 2 ;;
    --h-cycles) H_CYCLES="$2"; shift 2 ;;
    --l-cycles) L_CYCLES="$2"; shift 2 ;;
    --ema) EMA="$2"; shift 2 ;;

    --k-sweep) K_SWEEP="$2"; shift 2 ;;
    --python-bin) PYTHON_BIN="$2"; shift 2 ;;
    --torchrun-bin) TORCHRUN_BIN="$2"; shift 2 ;;
    --nproc-per-node) NPROC_PER_NODE="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *)
      echo bash scripts/run_spec_ptrm_ablations.sh "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x "./venv/bin/python" ]]; then
    PYTHON_BIN="./venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Could not find python executable. Use --python-bin." >&2
    exit 1
  fi
fi

if [[ ! -f "config/arch/ptrm.yaml" ]]; then
  echo "Missing config/arch/ptrm.yaml. Ensure PTRM is added first." >&2
  exit 1
fi

run_cmd() {
  local run_name="$1"
  shift

  local -a base_args=(
    "+project_name=${WANDB_PROJECT}"
    "+run_name=${run_name}"
    "data_paths=[${DATA_DIR}]"
    "evaluators=[]"
    "epochs=${EPOCHS}"
    "eval_interval=${EVAL_INTERVAL}"
    "global_batch_size=${GLOBAL_BATCH_SIZE}"
    "lr=${LR}"
    "puzzle_emb_lr=${PUZZLE_EMB_LR}"
    "weight_decay=${WEIGHT_DECAY}"
    "puzzle_emb_weight_decay=${PUZZLE_EMB_WEIGHT_DECAY}"
    "arch.L_layers=${L_LAYERS}"
    "arch.H_cycles=${H_CYCLES}"
    "arch.L_cycles=${L_CYCLES}"
    "ema=${EMA}"
  )

  echo "Starting ${run_name}"
  if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    "${TORCHRUN_BIN}" --nproc-per-node "${NPROC_PER_NODE}" \
      --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 \
      pretrain.py "${base_args[@]}" "$@"
  else
    "${PYTHON_BIN}" pretrain.py "${base_args[@]}" "$@"
  fi
}

if [[ "${SKIP_DATASET}" -eq 0 ]]; then
  echo "Building/downloading Maze dataset into ${DATA_DIR}"
  dataset_args=(
    dataset/build_maze_dataset.py
    --source-repo "${SOURCE_REPO}"
    --output-dir "${DATA_DIR}"
  )
  if [[ -n "${SUBSAMPLE_SIZE}" ]]; then
    dataset_args+=(--subsample-size "${SUBSAMPLE_SIZE}")
  fi
  if [[ "${AUG}" == "true" ]]; then
    dataset_args+=(--aug)
  else
    dataset_args+=(--no-aug)
  fi
  "${PYTHON_BIN}" "${dataset_args[@]}"
else
  echo "Skipping dataset build step"
fi

# Baseline TRM-Att short run
run_cmd "${RUN_PREFIX}_trm_att_maze" \
  "arch=trm"

# PTRM-v0 K sweep
for k in ${K_SWEEP}; do
  run_cmd "${RUN_PREFIX}_ptrm_maze_k${k}" \
    "arch=ptrm" \
    "arch.z_slots=${k}"
done

echo "All PTRM Maze ablation runs completed."
