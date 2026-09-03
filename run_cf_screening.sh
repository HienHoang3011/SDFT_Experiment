#!/usr/bin/env bash
set -euo pipefail

# One-command Qwen2-7B screening on the three strongest FAPM task candidates.
# Training uses the repository's existing train_sft.py and train_sft_lora.py.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-uv}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-7B}"
TASKS="${TASKS:-winogrande squad rte}"
DATA_ROOT="${DATA_ROOT:-data/cf_benchmarks}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/cf-screening}"
SHARED_REPORT_DIR="${SHARED_REPORT_DIR:-reports-cf-shared}"
KEEP_MODELS="${KEEP_MODELS:-0}"

# Disclosed by the FAPM paper: LR 1e-5, global batch 64, 3 epochs, AdamW.
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EPOCHS="${EPOCHS:-3}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"

# Optional small values are useful for a smoke test before the full run.
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
DOWNSTREAM_EVAL_LIMIT="${DOWNSTREAM_EVAL_LIMIT:-0}"
PRIOR_BATCH_SIZE="${PRIOR_BATCH_SIZE:-auto}"

run_python() {
  "$UV_BIN" run python "$@"
}

mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT_ABS="$(cd "$OUTPUT_ROOT" && pwd -P)"
if [[ "$OUTPUT_ROOT_ABS" == "/" ]]; then
  echo "Refusing to use / as OUTPUT_ROOT" >&2
  exit 2
fi

remove_model_dir() {
  local target="$1"
  if [[ "$KEEP_MODELS" == "1" ]]; then
    echo "[cleanup] KEEP_MODELS=1; keeping $target"
    return
  fi

  local target_parent target_abs
  target_parent="$(cd "$(dirname "$target")" && pwd -P)"
  target_abs="$target_parent/$(basename "$target")"
  case "$target_abs" in
    "$OUTPUT_ROOT_ABS"/*)
      rm -rf -- "$target_abs"
      echo "[cleanup] Removed evaluated temporary model: $target_abs"
      ;;
    *)
      echo "Refusing to delete model outside OUTPUT_ROOT: $target_abs" >&2
      exit 2
      ;;
  esac
}

read -r -a TASK_ARRAY <<< "$TASKS"

run_python prepare_cf_benchmarks.py \
  --tasks "${TASK_ARRAY[@]}" \
  --output_root "$DATA_ROOT" \
  --report_root "$ROOT_DIR" \
  --max_train_samples "$MAX_TRAIN_SAMPLES" \
  --max_eval_samples "$MAX_EVAL_SAMPLES"

# Prior capabilities are evaluated once for the common base model. The existing
# evaluator now intentionally excludes WinoGrande from every prior suite.
run_python eval/eval_previous_task.py \
  --model_path "$BASE_MODEL" \
  --backend hf \
  --batch_size "$PRIOR_BATCH_SIZE" \
  --output_dir "$SHARED_REPORT_DIR/base"

for task in "${TASK_ARRAY[@]}"; do
  echo
  echo "========== CF screening: $task =========="

  report_dir="reports-$task"
  full_model_dir="$OUTPUT_ROOT/$task/full"
  lora_model_dir="$OUTPUT_ROOT/$task/lora"
  train_data="$DATA_ROOT/$task/train"
  dev_data="$DATA_ROOT/$task/dev"

  mkdir -p "$report_dir/base"
  cp "$SHARED_REPORT_DIR/base/previous_capabilities_summary.json" \
    "$report_dir/base/previous_capabilities_summary.json"
  cp "$SHARED_REPORT_DIR/base/previous_capabilities_detailed.json" \
    "$report_dir/base/previous_capabilities_detailed.json"

  run_python eval/eval_cf_task.py \
    --task "$task" \
    --model_path "$BASE_MODEL" \
    --data_root "$DATA_ROOT" \
    --output_dir "$report_dir/base" \
    --limit "$DOWNSTREAM_EVAL_LIMIT"

  run_python methods/train_sft.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_data_path "$train_data" \
    --dev_data_path "$dev_data" \
    --output_dir "$full_model_dir" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$EPOCHS" \
    --max_seq_length "$MAX_SEQ_LENGTH"

  run_python eval/eval_cf_task.py \
    --task "$task" \
    --model_path "$full_model_dir" \
    --data_root "$DATA_ROOT" \
    --output_dir "$report_dir/full" \
    --limit "$DOWNSTREAM_EVAL_LIMIT"

  run_python eval/eval_previous_task.py \
    --model_path "$full_model_dir" \
    --backend hf \
    --batch_size "$PRIOR_BATCH_SIZE" \
    --output_dir "$report_dir/full"

  remove_model_dir "$full_model_dir"

  run_python methods/train_sft_lora.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_data_path "$train_data" \
    --dev_data_path "$dev_data" \
    --output_dir "$lora_model_dir" \
    --learning_rate "$LEARNING_RATE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --num_train_epochs "$EPOCHS" \
    --max_seq_length "$MAX_SEQ_LENGTH"

  run_python eval/eval_cf_task.py \
    --task "$task" \
    --model_path "$lora_model_dir" \
    --data_root "$DATA_ROOT" \
    --output_dir "$report_dir/lora" \
    --limit "$DOWNSTREAM_EVAL_LIMIT"

  run_python eval/eval_previous_task.py \
    --model_path "$lora_model_dir" \
    --backend hf \
    --batch_size "$PRIOR_BATCH_SIZE" \
    --output_dir "$report_dir/lora"

  remove_model_dir "$lora_model_dir"

  run_python eval/generate_cf_report.py \
    --task "$task" \
    --report_dir "$report_dir"
done

echo
echo "Completed. Reports: reports-winogrande, reports-squad, reports-rte"
