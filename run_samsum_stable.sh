#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-7B}"
RUN_NAME="${RUN_NAME:-fair-methods-2epoch}"
DATA_ROOT="${DATA_ROOT:-data/samsum_emnlp2019}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/samsum/$RUN_NAME}"
MODEL_ROOT="$ARTIFACT_ROOT/models"
REPORT_ROOT="$ARTIFACT_ROOT/reports"
BENCHMARK_ROOT="$ARTIFACT_ROOT/stable-five"

EPOCHS="${EPOCHS:-2}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
SEED="${SEED:-42}"
EWC_LAMBDA="${EWC_LAMBDA:-100}"
FISHER_MAX_SAMPLES="${FISHER_MAX_SAMPLES:-256}"
STEER_X_FACTOR="${STEER_X_FACTOR:-1}"
FAPM_K_RATIO="${FAPM_K_RATIO:-0.1}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python not executable: $PYTHON_BIN" >&2
  exit 2
fi
for target in "$MODEL_ROOT" "$REPORT_ROOT" "$BENCHMARK_ROOT"; do
  if [[ -e "$target" && -n "$(find "$target" -mindepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty output: $target" >&2
    exit 2
  fi
done
mkdir -p "$MODEL_ROOT" "$REPORT_ROOT"
run_python() { "$PYTHON_BIN" -u "$@"; }

if [[ ! -f "$DATA_ROOT/manifest.json" ]]; then
  run_python prepare_samsum_emnlp2019.py --output_root "$DATA_ROOT"
fi
run_python eval/write_samsum_manifest.py \
  --output "$REPORT_ROOT/fairness_manifest.json" \
  --run_name "$RUN_NAME" --epochs "$EPOCHS" --learning_rate "$LEARNING_RATE" \
  --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
  --max_sequence_length "$MAX_SEQ_LENGTH" --seed "$SEED" \
  --fisher_max_samples "$FISHER_MAX_SAMPLES" \
  --ewc_lambda "$EWC_LAMBDA" --steer_x_factor "$STEER_X_FACTOR" \
  --fapm_k_ratio "$FAPM_K_RATIO"

shared=(
  --model_name_or_path "$BASE_MODEL"
  --train_data_path "$DATA_ROOT/train"
  --dev_data_path "$DATA_ROOT/dev"
  --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs "$EPOCHS"
  --max_seq_length "$MAX_SEQ_LENGTH"
  --seed "$SEED"
)

run_python methods/train_sft.py "${shared[@]}" --output_dir "$MODEL_ROOT/full_sft"
run_python methods/train_sft_lora.py "${shared[@]}" --output_dir "$MODEL_ROOT/lora"
run_python methods/train_ewc.py "${shared[@]}" --output_dir "$MODEL_ROOT/ewc" \
  --fisher_max_samples "$FISHER_MAX_SAMPLES" --ewc_lambda "$EWC_LAMBDA"
run_python methods/train_steer.py "${shared[@]}" --output_dir "$MODEL_ROOT/steer_lora" \
  --x_factor "$STEER_X_FACTOR"
run_python methods/train_steer.py "${shared[@]}" --output_dir "$MODEL_ROOT/steer_full" \
  --x_factor "$STEER_X_FACTOR" --disable_peft
run_python methods/fapm.py --base_model "$BASE_MODEL" --sft_model "$MODEL_ROOT/full_sft" \
  --output_dir "$MODEL_ROOT/fapm" --k_ratio "$FAPM_K_RATIO"

for name in base full_sft lora ewc steer_lora steer_full fapm; do
  model_path="$MODEL_ROOT/$name"
  if [[ "$name" == base ]]; then model_path="$BASE_MODEL"; fi
  run_python eval/eval_samsum.py --model_path "$model_path" --data_root "$DATA_ROOT" \
    --output_dir "$REPORT_ROOT/$name"
done

run_python benchmarks/stable_five/run.py --output-dir "$BENCHMARK_ROOT" \
  --model-path "base=$BASE_MODEL" \
  --model-path "full_sft=$MODEL_ROOT/full_sft" \
  --model-path "lora=$MODEL_ROOT/lora" \
  --model-path "ewc=$MODEL_ROOT/ewc" \
  --model-path "steer_lora=$MODEL_ROOT/steer_lora" \
  --model-path "steer_full=$MODEL_ROOT/steer_full" \
  --model-path "fapm=$MODEL_ROOT/fapm"
echo "Fair method comparison completed: $ARTIFACT_ROOT"
