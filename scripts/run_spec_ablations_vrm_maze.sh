#!/usr/bin/env bash
set -euo pipefail

WANDB_PROJECT="MazeHard-VRM-Ablations"
DATA_DIR="data/maze-30x30-hard-1k"
EPOCHS=50000
EVAL_INTERVAL=5000
LR=1e-4
PUZZLE_EMB_LR=1e-4
WEIGHT_DECAY=1.0
PUZZLE_EMB_WEIGHT_DECAY=1.0
GLOBAL_BATCH_SIZE=768
L_LAYERS=2
H_CYCLES=3
L_CYCLES=4
EMA=true
SKIP_DATASET=0
NUM_GPUS=1
MAZE_AUG=false
SUBSAMPLE_SIZE=""
RUN_SELECTOR="all"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_spec_ablations_vrm_maze.sh [options]

Options:
  --wandb-project <name>       W&B project name (default: MazeHard-VRM-Ablations)
  --data-dir <path>            Dataset directory (default: data/maze-30x30-hard-1k)
  --epochs <int>               Training epochs per run (default: 50000)
  --eval-interval <int>        Evaluation interval (default: 5000)
  --lr <float>                 Model lr (default: 1e-4)
  --puzzle-emb-lr <float>      Puzzle embedding lr (default: 1e-4)
  --weight-decay <float>       Model weight decay (default: 1.0)
  --puzzle-emb-weight-decay <float> Puzzle embedding weight decay (default: 1.0)
  --global-batch-size <int>    Global batch size (default: 768)
  --l-layers <int>             arch.L_layers (default: 2)
  --h-cycles <int>             arch.H_cycles (default: 3)
  --l-cycles <int>             arch.L_cycles (default: 4)
  --ema <true|false>           EMA toggle (default: true)
  --num-gpus <int>             Number of GPUs; uses torchrun when >1 (default: 1)
  --nproc-per-node <int>       Alias for --num-gpus
  --maze-aug <true|false>      Build maze with dihedral aug (default: false)
  --subsample-size <int>       Optional maze train subsample for dataset build
  --run <id|list|all>          Run selection: 1, 2, 3, 1,3, or all (default: all)
  --skip-dataset               Skip dataset build/download step
  --help                       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --eval-interval) EVAL_INTERVAL="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --puzzle-emb-lr) PUZZLE_EMB_LR="$2"; shift 2 ;;
    --weight-decay) WEIGHT_DECAY="$2"; shift 2 ;;
    --puzzle-emb-weight-decay) PUZZLE_EMB_WEIGHT_DECAY="$2"; shift 2 ;;
    --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
    --l-layers) L_LAYERS="$2"; shift 2 ;;
    --h-cycles) H_CYCLES="$2"; shift 2 ;;
    --l-cycles) L_CYCLES="$2"; shift 2 ;;
    --ema) EMA="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --nproc-per-node) NUM_GPUS="$2"; shift 2 ;;
    --maze-aug) MAZE_AUG="$2"; shift 2 ;;
    --subsample-size) SUBSAMPLE_SIZE="$2"; shift 2 ;;
    --run) RUN_SELECTOR="$2"; shift 2 ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

RUN_IDS=()
if [[ "${RUN_SELECTOR}" == "all" ]]; then
  RUN_IDS=(1 2 3)
else
  RUN_SPEC="${RUN_SELECTOR//,/ }"
  read -r -a RUN_IDS <<< "${RUN_SPEC}"
  if [[ "${#RUN_IDS[@]}" -eq 0 ]]; then
    echo "Invalid --run value: ${RUN_SELECTOR}. Use 1, 2, 3, 1,3, or all." >&2
    exit 1
  fi
  for run_id in "${RUN_IDS[@]}"; do
    if ! [[ "${run_id}" =~ ^[0-9]+$ ]]; then
      echo "Invalid --run value: ${RUN_SELECTOR}. Use numeric run ids." >&2
      exit 1
    fi
    if [[ "${run_id}" -lt 1 || "${run_id}" -gt 3 ]]; then
      echo "Unknown run id ${run_id}. Valid ids are: 1, 2, 3." >&2
      exit 1
    fi
  done
fi

if [[ "${SKIP_DATASET}" -eq 0 ]]; then
  echo "Building/downloading Maze-Hard dataset into ${DATA_DIR}"
  BUILD_CMD=(python dataset/build_maze_dataset.py --output-dir "${DATA_DIR}")
  case "${MAZE_AUG}" in
    true|True|TRUE)
      BUILD_CMD+=(--aug)
      ;;
    false|False|FALSE)
      BUILD_CMD+=(--no-aug)
      ;;
    *)
      echo "Invalid --maze-aug value: ${MAZE_AUG}. Use true or false." >&2
      exit 1
      ;;
  esac
  if [[ -n "${SUBSAMPLE_SIZE}" ]]; then
    BUILD_CMD+=(--subsample-size "${SUBSAMPLE_SIZE}")
  fi
  "${BUILD_CMD[@]}"
else
  echo "Skipping dataset build step"
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  PRETRAIN_CMD=(torchrun --nproc-per-node "${NUM_GPUS}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py)
else
  PRETRAIN_CMD=(python pretrain.py)
fi

run_pretrain() {
  local run_name="$1"
  shift

  echo "Starting ${run_name}"
  "${PRETRAIN_CMD[@]}" \
    "+project_name=${WANDB_PROJECT}" \
    "+run_name=${run_name}" \
    "data_paths=[${DATA_DIR}]" \
    "evaluators=[]" \
    "epochs=${EPOCHS}" \
    "eval_interval=${EVAL_INTERVAL}" \
    "lr=${LR}" \
    "puzzle_emb_lr=${PUZZLE_EMB_LR}" \
    "weight_decay=${WEIGHT_DECAY}" \
    "puzzle_emb_weight_decay=${PUZZLE_EMB_WEIGHT_DECAY}" \
    "global_batch_size=${GLOBAL_BATCH_SIZE}" \
    "arch.L_layers=${L_LAYERS}" \
    "arch.H_cycles=${H_CYCLES}" \
    "arch.L_cycles=${L_CYCLES}" \
    "ema=${EMA}" \
    "$@"
}

should_run() {
  local target_id="$1"
  for run_id in "${RUN_IDS[@]}"; do
    if [[ "${run_id}" == "${target_id}" ]]; then
      return 0
    fi
  done
  return 1
}

# 1) Baseline TRM-Att
if should_run 1; then
  run_pretrain "pretrain_att_maze30x30_spec_vrm" \
    "arch=trm"
fi

# 2) VRM-Axial
if should_run 2; then
  run_pretrain "pretrain_axial_maze30x30_spec_vrm" \
    "arch=vrm" \
    "arch.axial_t=True" \
    "arch.pos_encodings=none"
fi

# 3) VRM-Axial + 2dconv
if should_run 3; then
  run_pretrain "pretrain_axial_2dconv_maze30x30_spec_vrm" \
    "arch=vrm" \
    "arch.axial_t=True" \
    "arch.axial_2dconv=True" \
    "arch.axial_2dconv_kernel=3" \
    "arch.pos_encodings=none"
fi

echo "Maze-Hard ablations submitted."
