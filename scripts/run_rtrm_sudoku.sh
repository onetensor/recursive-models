#!/usr/bin/env bash
set -euo pipefail

WANDB_PROJECT="Sudoku-RTRM"
DATA_DIR="data/sudoku-extreme-1k-aug-1000"
SOURCE_REPO="sapientinc/sudoku-extreme"
RUN_NAME="pretrain_att_sudoku_rtrm"

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
NUM_GPUS=1
GLOBAL_BATCH_SIZE=""

SUBSAMPLE_SIZE=1000
NUM_AUG=1000
SKIP_DATASET=0

# R-TRM options (enabled by default).
RESIDUAL_ENABLED=true
RESIDUAL_TRACE_ENABLED=true
HALT_RESIDUAL_ENABLED=true
HALT_RESIDUAL_STAT="max"
RESIDUAL_TYPE="logits_l2"
RESIDUAL_TEMP=1.0
HALT_RESIDUAL_TAU=1e-3
HALT_RESIDUAL_PATIENCE=2
HALT_RESIDUAL_MIN_STEPS=2
HALT_CONFIDENCE_MIN=0.0
HALT_CONFIDENCE_STAT="min"
HALT_CONFIDENCE_TEMP=1.0
STICKINESS_WEIGHT=0.0
STICKINESS_HORIZON=1
STICKINESS_GAMMA=1.0
UPDATE_DAMPING_ENABLED=false
UPDATE_DAMPING_ALPHA_ZL=1.0
UPDATE_DAMPING_ALPHA_ZH=1.0
EVAL_SAVE_OUTPUTS="[residual_trace_max,residual_trace_mean,confidence_trace_min,confidence_trace_mean]"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_rtrm_sudoku.sh [options]

Options:
  --wandb-project <name>       W&B project name (default: Sudoku-RTRM)
  --data-dir <path>            Dataset directory (default: data/sudoku-extreme-1k-aug-1000)
  --source-repo <repo>         HF dataset repo for Sudoku build (default: sapientinc/sudoku-extreme)
  --run-name <name>            Run name (default: pretrain_att_sudoku_rtrm)
  --epochs <int>               Training epochs (default: 50000)
  --eval-interval <int>        Evaluation interval (default: 5000)
  --global-batch-size <int>    Optional global batch size override
  --lr <float>                 Model lr (default: 1e-4)
  --puzzle-emb-lr <float>      Puzzle embedding lr (default: 1e-4)
  --weight-decay <float>       Model weight decay (default: 1.0)
  --puzzle-emb-weight-decay <float> Puzzle embedding weight decay (default: 1.0)
  --l-layers <int>             arch.L_layers (default: 2)
  --h-cycles <int>             arch.H_cycles (default: 3)
  --l-cycles <int>             arch.L_cycles (default: 6)
  --ema <true|false>           EMA toggle (default: true)
  --num-gpus <int>             Number of GPUs; uses torchrun when >1 (default: 1)
  --nproc-per-node <int>       Alias for --num-gpus

  --subsample-size <int>       Sudoku train subsample size for dataset build (default: 1000)
  --num-aug <int>              Sudoku augment count for dataset build (default: 1000)
  --skip-dataset               Skip dataset build check/fetch step

  --residual-enabled <bool>        arch.residual_enabled (default: true)
  --residual-trace-enabled <bool>  arch.residual_trace_enabled (default: true)
  --halt-residual-enabled <bool>   arch.halt_residual_enabled (default: true)
  --halt-residual-stat <mean|max>  arch.halt_residual_stat (default: max)
  --residual-type <type>           arch.residual_type (default: logits_l2)
  --residual-temp <float>          arch.residual_temp (default: 1.0)
  --halt-residual-tau <float>      arch.halt_residual_tau (default: 1e-3)
  --halt-residual-patience <int>   arch.halt_residual_patience (default: 2)
  --halt-residual-min-steps <int>  arch.halt_residual_min_steps (default: 2)
  --halt-confidence-min <float>    arch.halt_confidence_min (default: 0.0)
  --halt-confidence-stat <mean|min> arch.halt_confidence_stat (default: min)
  --halt-confidence-temp <float>   arch.halt_confidence_temp (default: 1.0)
  --stickiness-weight <float>      arch.loss.stickiness_weight (default: 0.0)
  --stickiness-horizon <int>       arch.loss.stickiness_horizon (default: 1)
  --stickiness-gamma <float>       arch.loss.stickiness_gamma (default: 1.0)
  --update-damping-enabled <bool>  arch.update_damping_enabled (default: false)
  --update-damping-alpha-zl <float> arch.update_damping_alpha_zL (default: 1.0)
  --update-damping-alpha-zh <float> arch.update_damping_alpha_zH (default: 1.0)
  --eval-save-outputs <list>       eval_save_outputs list string

  --help                       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --data-dir) DATA_DIR="$2"; shift 2 ;;
    --source-repo) SOURCE_REPO="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
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
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --nproc-per-node) NUM_GPUS="$2"; shift 2 ;;
    --subsample-size) SUBSAMPLE_SIZE="$2"; shift 2 ;;
    --num-aug) NUM_AUG="$2"; shift 2 ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --residual-enabled) RESIDUAL_ENABLED="$2"; shift 2 ;;
    --residual-trace-enabled) RESIDUAL_TRACE_ENABLED="$2"; shift 2 ;;
    --halt-residual-enabled) HALT_RESIDUAL_ENABLED="$2"; shift 2 ;;
    --halt-residual-stat) HALT_RESIDUAL_STAT="$2"; shift 2 ;;
    --residual-type) RESIDUAL_TYPE="$2"; shift 2 ;;
    --residual-temp) RESIDUAL_TEMP="$2"; shift 2 ;;
    --halt-residual-tau) HALT_RESIDUAL_TAU="$2"; shift 2 ;;
    --halt-residual-patience) HALT_RESIDUAL_PATIENCE="$2"; shift 2 ;;
    --halt-residual-min-steps) HALT_RESIDUAL_MIN_STEPS="$2"; shift 2 ;;
    --halt-confidence-min) HALT_CONFIDENCE_MIN="$2"; shift 2 ;;
    --halt-confidence-stat) HALT_CONFIDENCE_STAT="$2"; shift 2 ;;
    --halt-confidence-temp) HALT_CONFIDENCE_TEMP="$2"; shift 2 ;;
    --stickiness-weight) STICKINESS_WEIGHT="$2"; shift 2 ;;
    --stickiness-horizon) STICKINESS_HORIZON="$2"; shift 2 ;;
    --stickiness-gamma) STICKINESS_GAMMA="$2"; shift 2 ;;
    --update-damping-enabled) UPDATE_DAMPING_ENABLED="$2"; shift 2 ;;
    --update-damping-alpha-zl) UPDATE_DAMPING_ALPHA_ZL="$2"; shift 2 ;;
    --update-damping-alpha-zh) UPDATE_DAMPING_ALPHA_ZH="$2"; shift 2 ;;
    --eval-save-outputs) EVAL_SAVE_OUTPUTS="$2"; shift 2 ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

dataset_exists() {
  local required_files=(
    "${DATA_DIR}/train/dataset.json"
    "${DATA_DIR}/test/dataset.json"
    "${DATA_DIR}/train/all__inputs.npy"
    "${DATA_DIR}/train/all__labels.npy"
    "${DATA_DIR}/test/all__inputs.npy"
    "${DATA_DIR}/test/all__labels.npy"
  )
  local f
  for f in "${required_files[@]}"; do
    if [[ ! -f "${f}" ]]; then
      return 1
    fi
  done
  return 0
}

if [[ "${SKIP_DATASET}" -eq 0 ]]; then
  if dataset_exists; then
    echo "Dataset already exists at ${DATA_DIR}; skipping build/download."
  else
    echo "Dataset not found at ${DATA_DIR}; building/downloading Sudoku-Extreme."
    python dataset/build_sudoku_dataset.py \
      --source-repo "${SOURCE_REPO}" \
      --output-dir "${DATA_DIR}" \
      --subsample-size "${SUBSAMPLE_SIZE}" \
      --num-aug "${NUM_AUG}"
  fi
else
  echo "Skipping dataset build step"
fi

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  PRETRAIN_CMD=(torchrun --nproc-per-node "${NUM_GPUS}" --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nnodes=1 pretrain.py)
else
  PRETRAIN_CMD=(python pretrain.py)
fi

CMD=(
  "${PRETRAIN_CMD[@]}"
  "+project_name=${WANDB_PROJECT}"
  "+run_name=${RUN_NAME}"
  "arch=rtrm"
  "data_paths=[${DATA_DIR}]"
  "evaluators=[]"
  "epochs=${EPOCHS}"
  "eval_interval=${EVAL_INTERVAL}"
  "eval_save_outputs=${EVAL_SAVE_OUTPUTS}"
  "lr=${LR}"
  "puzzle_emb_lr=${PUZZLE_EMB_LR}"
  "weight_decay=${WEIGHT_DECAY}"
  "puzzle_emb_weight_decay=${PUZZLE_EMB_WEIGHT_DECAY}"
  "arch.L_layers=${L_LAYERS}"
  "arch.H_cycles=${H_CYCLES}"
  "arch.L_cycles=${L_CYCLES}"
  "ema=${EMA}"
  "arch.residual_enabled=${RESIDUAL_ENABLED}"
  "arch.residual_trace_enabled=${RESIDUAL_TRACE_ENABLED}"
  "arch.halt_residual_enabled=${HALT_RESIDUAL_ENABLED}"
  "arch.halt_residual_stat=${HALT_RESIDUAL_STAT}"
  "arch.residual_type=${RESIDUAL_TYPE}"
  "arch.residual_temp=${RESIDUAL_TEMP}"
  "arch.halt_residual_tau=${HALT_RESIDUAL_TAU}"
  "arch.halt_residual_patience=${HALT_RESIDUAL_PATIENCE}"
  "arch.halt_residual_min_steps=${HALT_RESIDUAL_MIN_STEPS}"
  "arch.halt_confidence_min=${HALT_CONFIDENCE_MIN}"
  "arch.halt_confidence_stat=${HALT_CONFIDENCE_STAT}"
  "arch.halt_confidence_temp=${HALT_CONFIDENCE_TEMP}"
  "arch.loss.stickiness_weight=${STICKINESS_WEIGHT}"
  "arch.loss.stickiness_horizon=${STICKINESS_HORIZON}"
  "arch.loss.stickiness_gamma=${STICKINESS_GAMMA}"
  "arch.update_damping_enabled=${UPDATE_DAMPING_ENABLED}"
  "arch.update_damping_alpha_zL=${UPDATE_DAMPING_ALPHA_ZL}"
  "arch.update_damping_alpha_zH=${UPDATE_DAMPING_ALPHA_ZH}"
)

if [[ -n "${GLOBAL_BATCH_SIZE}" ]]; then
  CMD+=("global_batch_size=${GLOBAL_BATCH_SIZE}")
fi

echo "Starting ${RUN_NAME}"
"${CMD[@]}"
