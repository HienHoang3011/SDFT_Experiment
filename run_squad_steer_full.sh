#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

UV_BIN="${UV_BIN:-uv}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-7B}"
DATA_ROOT="${DATA_ROOT:-data/cf_benchmarks}"
MODEL_DIR="${MODEL_DIR:-$ROOT_DIR/outputs/steer-full-squad-qwen2-7b}"
SQUAD_EVAL_DIR="${SQUAD_EVAL_DIR:-$ROOT_DIR/outputs/steer_full_squad_eval}"
PRIOR_EVAL_DIR="${PRIOR_EVAL_DIR:-$ROOT_DIR/outputs/steer_full_prior_eval}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports-squad}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT_DIR/evals_and_steer_images/squad_analysis}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/logs/squad_steer_full.log}"

LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EPOCHS="${EPOCHS:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-64}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-2048}"
PRIOR_BATCH_SIZE="${PRIOR_BATCH_SIZE:-4}"
KL_PROBE_SAMPLES="${KL_PROBE_SAMPLES:-250}"
KL_PROBE_MAX_TOKENS_PER_BATCH="${KL_PROBE_MAX_TOKENS_PER_BATCH:-128}"

mkdir -p "$ROOT_DIR/logs" "$SQUAD_EVAL_DIR" "$PRIOR_EVAL_DIR" "$REPORT_DIR"

outputs_root_abs="$(cd "$ROOT_DIR/outputs" && pwd -P)"
model_parent="$(dirname "$MODEL_DIR")"
mkdir -p "$model_parent"
model_parent_abs="$(cd "$model_parent" && pwd -P)"
model_dir_abs="$model_parent_abs/$(basename "$MODEL_DIR")"
case "$model_dir_abs" in
  "$outputs_root_abs"/*) ;;
  *)
    echo "Refusing to use a temporary model directory outside $outputs_root_abs" >&2
    exit 2
    ;;
esac

if [[ -d "$model_dir_abs" ]] && [[ -n "$(find "$model_dir_abs" -mindepth 1 -print -quit)" ]]; then
  echo "Refusing to overwrite non-empty model directory: $model_dir_abs" >&2
  echo "Set MODEL_DIR to a fresh path or move the existing checkpoint first." >&2
  exit 2
fi

run_python() {
  "$UV_BIN" run python -u "$@"
}

run_pipeline() {
  run_python methods/train_steer.py \
    --model_name_or_path "$BASE_MODEL" \
    --train_data_path "$DATA_ROOT/squad/train" \
    --eval_data_path "$DATA_ROOT/squad/dev" \
    --dev_data_path "$DATA_ROOT/squad/dev" \
    --output_dir "$model_dir_abs" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$EPOCHS" \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --kl_probe_samples "$KL_PROBE_SAMPLES" \
    --kl_probe_max_tokens_per_batch "$KL_PROBE_MAX_TOKENS_PER_BATCH" \
    --disable_peft

  run_python eval/eval_cf_task.py \
    --task squad \
    --model_path "$model_dir_abs" \
    --data_root "$DATA_ROOT" \
    --output_dir "$SQUAD_EVAL_DIR"

  run_python eval/eval_prior_capabilities.py \
    --model_path "$model_dir_abs" \
    --batch_size "$PRIOR_BATCH_SIZE" \
    --output_dir "$PRIOR_EVAL_DIR"

  run_python eval/generate_report.py \
    --base_new_task outputs/base_squad_eval/downstream_results.json \
    --base_prior_task outputs/base_prior_eval/summary.json \
    --sft_new_task outputs/sft_squad_eval/downstream_results.json \
    --sft_prior_task outputs/sft_prior_eval/summary.json \
    --sft_lora_new_task outputs/sft_lora_squad_eval/downstream_results.json \
    --sft_lora_prior_task outputs/sft_lora_prior_eval/summary.json \
    --steer_full_new_task "$SQUAD_EVAL_DIR/downstream_results.json" \
    --steer_full_prior_task "$PRIOR_EVAL_DIR/summary.json" \
    --output_dir "$REPORT_DIR"
}

set +e
run_pipeline 2>&1 | tee -a "$LOG_PATH"
pipeline_status="${PIPESTATUS[0]}"
set -e

ANALYSIS_DIR="$ANALYSIS_DIR" \
REPORT_DIR="$REPORT_DIR" \
STEER_MODEL_DIR="$model_dir_abs" \
  "$ROOT_DIR/collect_squad_analysis.sh" 2>&1 | tee -a "$LOG_PATH"

if [[ "$pipeline_status" -ne 0 ]]; then
  echo "Steer-full pipeline failed with status $pipeline_status; checkpoint kept for debugging." \
    | tee -a "$LOG_PATH"
  exit "$pipeline_status"
fi

rm -rf -- "$model_dir_abs"
echo "Removed evaluated temporary model: $model_dir_abs" | tee -a "$LOG_PATH"
echo "Steer-full SQuAD pipeline completed successfully." | tee -a "$LOG_PATH"
