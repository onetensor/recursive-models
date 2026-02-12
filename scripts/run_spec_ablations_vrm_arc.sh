#!/usr/bin/env bash
set -euo pipefail

WANDB_PROJECT="ARC-VRM-Ablations"
INPUT_FILE_PREFIX="kaggle/combined/arc-agi"
ARC_TARGET="arc1" # arc1 | arc2 | both
DATA_DIR_ARC1="data/arc1concept-aug-1000"
DATA_DIR_ARC2="data/arc2concept-aug-1000"
EPOCHS=100000
EVAL_INTERVAL=10000
LR=1e-4
PUZZLE_EMB_LR=1e-4
WEIGHT_DECAY=0.1
PUZZLE_EMB_WEIGHT_DECAY=0.1
GLOBAL_BATCH_SIZE=768
L_LAYERS=2
H_CYCLES=3
L_CYCLES=4
EMA=true
SKIP_DATASET=0
NUM_GPUS=1
DISABLE_EVALUATORS=0
NUM_AUG=1000
RUN_SELECTOR="all"

usage() {
  cat <<'EOF'
Usage: bash scripts/run_spec_ablations_vrm_arc.sh [options]

Options:
  --wandb-project <name>       W&B project name (default: ARC-VRM-Ablations)
  --input-file-prefix <path>   ARC json prefix (default: kaggle/combined/arc-agi)
  --arc-target <arc1|arc2|both> Dataset target (default: arc1)
  --data-dir-arc1 <path>       ARC-1 output dir (default: data/arc1concept-aug-1000)
  --data-dir-arc2 <path>       ARC-2 output dir (default: data/arc2concept-aug-1000)
  --epochs <int>               Training epochs per run (default: 100000)
  --eval-interval <int>        Evaluation interval (default: 10000)
  --lr <float>                 Model lr (default: 1e-4)
  --puzzle-emb-lr <float>      Puzzle embedding lr (default: 1e-4)
  --weight-decay <float>       Model weight decay (default: 0.1)
  --puzzle-emb-weight-decay <float> Puzzle embedding weight decay (default: 0.1)
  --global-batch-size <int>    Global batch size (default: 768)
  --l-layers <int>             arch.L_layers (default: 2)
  --h-cycles <int>             arch.H_cycles (default: 3)
  --l-cycles <int>             arch.L_cycles (default: 4)
  --ema <true|false>           EMA toggle (default: true)
  --num-gpus <int>             Number of GPUs; uses torchrun when >1 (default: 1)
  --nproc-per-node <int>       Alias for --num-gpus
  --num-aug <int>              ARC augmentation count for dataset build (default: 1000)
  --run <id|list|all>          Run selection: 1, 2, 3, 1,3, or all (default: all)
  --disable-evaluators         Force evaluators=[]
  --skip-dataset               Skip dataset build step
  --help                       Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
    --input-file-prefix) INPUT_FILE_PREFIX="$2"; shift 2 ;;
    --arc-target) ARC_TARGET="$2"; shift 2 ;;
    --data-dir-arc1) DATA_DIR_ARC1="$2"; shift 2 ;;
    --data-dir-arc2) DATA_DIR_ARC2="$2"; shift 2 ;;
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
    --num-aug) NUM_AUG="$2"; shift 2 ;;
    --run) RUN_SELECTOR="$2"; shift 2 ;;
    --disable-evaluators) DISABLE_EVALUATORS=1; shift ;;
    --skip-dataset) SKIP_DATASET=1; shift ;;
    --help) usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${ARC_TARGET}" != "arc1" && "${ARC_TARGET}" != "arc2" && "${ARC_TARGET}" != "both" ]]; then
  echo "Invalid --arc-target: ${ARC_TARGET}. Must be arc1, arc2, or both." >&2
  exit 1
fi

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

build_arc_dataset() {
  local subset_name="$1"
  local output_dir="$2"
  local subsets="$3"
  local test_set_name="$4"

  echo "Building ${subset_name} into ${output_dir}"
  python -m dataset.build_arc_dataset \
    --input-file-prefix "${INPUT_FILE_PREFIX}" \
    --output-dir "${output_dir}" \
    --subsets ${subsets} \
    --test-set-name "${test_set_name}" \
    --num-aug "${NUM_AUG}"
}

if [[ "${SKIP_DATASET}" -eq 0 ]]; then
  if [[ "${ARC_TARGET}" == "arc1" || "${ARC_TARGET}" == "both" ]]; then
    build_arc_dataset "ARC-AGI-1" "${DATA_DIR_ARC1}" "training evaluation concept" "evaluation"
  fi
  if [[ "${ARC_TARGET}" == "arc2" || "${ARC_TARGET}" == "both" ]]; then
    build_arc_dataset "ARC-AGI-2" "${DATA_DIR_ARC2}" "training2 evaluation2 concept" "evaluation2"
  fi
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
  local data_dir="$2"
  shift 2

  local cmd=(
    "${PRETRAIN_CMD[@]}"
    "+project_name=${WANDB_PROJECT}"
    "+run_name=${run_name}"
    "data_paths=[${data_dir}]"
    "epochs=${EPOCHS}"
    "eval_interval=${EVAL_INTERVAL}"
    "lr=${LR}"
    "puzzle_emb_lr=${PUZZLE_EMB_LR}"
    "weight_decay=${WEIGHT_DECAY}"
    "puzzle_emb_weight_decay=${PUZZLE_EMB_WEIGHT_DECAY}"
    "global_batch_size=${GLOBAL_BATCH_SIZE}"
    "arch.L_layers=${L_LAYERS}"
    "arch.H_cycles=${H_CYCLES}"
    "arch.L_cycles=${L_CYCLES}"
    "ema=${EMA}"
  )

  if [[ "${DISABLE_EVALUATORS}" -eq 1 ]]; then
    cmd+=("evaluators=[]")
  fi

  cmd+=("$@")

  echo "Starting ${run_name}"
  "${cmd[@]}"
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

run_target() {
  local tag="$1"
  local data_dir="$2"

  if should_run 1; then
    run_pretrain "pretrain_att_${tag}_spec_vrm" "${data_dir}" \
      "arch=trm"
  fi

  if should_run 2; then
    run_pretrain "pretrain_axial_${tag}_spec_vrm" "${data_dir}" \
      "arch=vrm" \
      "arch.axial_t=True" \
      "arch.pos_encodings=none"
  fi

  if should_run 3; then
    run_pretrain "pretrain_axial_2dconv_${tag}_spec_vrm" "${data_dir}" \
      "arch=vrm" \
      "arch.axial_t=True" \
      "arch.axial_2dconv=True" \
      "arch.axial_2dconv_kernel=3" \
      "arch.pos_encodings=none"
  fi
}

if [[ "${ARC_TARGET}" == "arc1" || "${ARC_TARGET}" == "both" ]]; then
  run_target "arc1concept" "${DATA_DIR_ARC1}"
fi

if [[ "${ARC_TARGET}" == "arc2" || "${ARC_TARGET}" == "both" ]]; then
  run_target "arc2concept" "${DATA_DIR_ARC2}"
fi

echo "ARC ablations submitted."
