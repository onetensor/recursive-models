#!/usr/bin/env bash
set -euo pipefail

WANDB_PROJECT="Sudoku-VRM-Ablations"
DATA_DIR="data/sudoku-extreme-1k-aug-1000"
EPOCHS=50000
EVAL_INTERVAL=5000
LR=1e-4
PUZZLE_EMB_LR=1e-4
WEIGHT_DECAY=1.0
PUZZLE_EMB_WEIGHT_DECAY=1.0
L_LAYERS=2
H_CYCLES=3
L_CYCLES=6
EMA=true
SUBSAMPLE_SIZE=1000
NUM_AUG=1000
SKIP_DATASET=0
NUM_GPUS=1

usage() {
  cat <<'EOF'
Usage: bash scripts/run_spec_ablations_vrm.sh [options]

Options:
  --wandb-project <name>       W&B project name (default: Sudoku-VRM-Ablations)
  --data-dir <path>            Dataset directory (default: data/sudoku-extreme-1k-aug-1000)
  --epochs <int>               Training epochs per run (default: 50000)
  --eval-interval <int>        Evaluation interval (default: 5000)
  --lr <float>                 Model lr (default: 1e-4)
  --puzzle-emb-lr <float>      Puzzle embedding lr (default: 1e-4)
  --weight-decay <float>       Model weight decay (default: 1.0)
  --puzzle-emb-weight-decay <float> Puzzle embedding weight decay (default: 1.0)
  --l-layers <int>             arch.L_layers (default: 2)
  --h-cycles <int>             arch.H_cycles (default: 3)
  --l-cycles <int>             arch.L_cycles (default: 6)
  --ema <true|false>           EMA toggle (default: true)
  --subsample-size <int>       Sudoku train subsample size for dataset build (default: 1000)
  --num-aug <int>              Sudoku augment count for dataset build (default: 1000)
  --num-gpus <int>             Number of GPUs; uses torchrun when >1 (default: 1)
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
    --l-layers) L_LAYERS="$2"; shift 2 ;;
    --h-cycles) H_CYCLES="$2"; shift 2 ;;
    --l-cycles) L_CYCLES="$2"; shift 2 ;;
    --ema) EMA="$2"; shift 2 ;;
    --subsample-size) SUBSAMPLE_SIZE="$2"; shift 2 ;;
    --num-aug) NUM_AUG="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${SKIP_DATASET}" -eq 0 ]]; then
  echo "Building/downloading Sudoku-Extreme dataset into ${DATA_DIR}"
  python dataset/build_sudoku_dataset.py \
    --output-dir "${DATA_DIR}" \
    --subsample-size "${SUBSAMPLE_SIZE}" \
    --num-aug "${NUM_AUG}"
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
    "arch.L_layers=${L_LAYERS}" \
    "arch.H_cycles=${H_CYCLES}" \
    "arch.L_cycles=${L_CYCLES}" \
    "ema=${EMA}" \
    "$@"
}

# 1) Baseline TRM-Att
run_pretrain "pretrain_att_sudoku_spec_vrm" \
  "arch=trm"

# 2) Baseline TRM-MLP
run_pretrain "pretrain_mlp_t_sudoku_spec_vrm" \
  "arch=trm" \
  "arch.mlp_t=True" \
  "arch.pos_encodings=none"

# 3) TRM-Axial (no conv)
run_pretrain "pretrain_axial_sudoku_spec_vrm" \
  "arch=vrm" \
  "arch.axial_t=True" \
  "arch.pos_encodings=none"

# 4) TRM-Axial + Axial2DConv
run_pretrain "pretrain_axial_2dconv_sudoku_spec_vrm" \
  "arch=vrm" \
  "arch.axial_t=True" \
  "arch.axial_2dconv=True" \
  "arch.axial_2dconv_kernel=3" \
  "arch.pos_encodings=none"

echo "All spec ablations submitted."
