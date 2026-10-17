#!/usr/bin/env bash
# Rerun the full-parameter methods with fp32 master weights (bf16 autocast compute).
# The original Full SFT / Steer-Full stored weights in bf16, so lr=1e-5 AdamW steps on
# |w| >= 2^-8 rounded away; FAPM post-processes Full SFT and is rerun with it.
# LoRA, Steer-LoRA and Base are unaffected and are reused from REFERENCE_ROOT.
# Every step is skipped when its output exists, so the script can be resumed.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/root/sdft-fp32-runs/venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-artifacts/base/Qwen2-7B}"
REFERENCE_ROOT="${REFERENCE_ROOT:-artifacts/samsum/fair-methods-2epoch-log25-tmux}"
RUN_NAME="${RUN_NAME:-fp32-master-2epoch-20261012}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-artifacts/samsum/$RUN_NAME}"
# Checkpoints are large (fp32 Full SFT/Steer-Full ≈ 30 GB each) and live outside /data.
MODEL_ROOT="${MODEL_ROOT:-/root/sdft-fp32-runs/models}"
OPTIM="${OPTIM:-adamw_8bit}"
REPORT_ROOT="$ARTIFACT_ROOT/reports"
LOG_DIR="$ARTIFACT_ROOT/logs"
mkdir -p "$MODEL_ROOT" "$REPORT_ROOT" "$LOG_DIR"
export PYTHONPATH="$ROOT_DIR/methods${PYTHONPATH:+:$PYTHONPATH}"

trap 'status=$?; printf "%s\n" "$status" > "$LOG_DIR/exit-status.txt"' EXIT
phase() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" | tee -a "$LOG_DIR/phases.log"; }
done_model() { [[ -f "$1/config.json" && -n "$(find "$1" -maxdepth 1 -name '*.safetensors' -print -quit)" ]]; }

if [[ ! -f "$LOG_DIR/environment.txt" ]]; then
  "$PYTHON_BIN" -m pip freeze > "$LOG_DIR/environment.txt"
fi
"$PYTHON_BIN" eval/write_samsum_manifest.py --output "$REPORT_ROOT/fairness_manifest.json" \
  --run_name "$RUN_NAME" --epochs 2 --learning_rate 1e-5 --per_device_batch_size 1 \
  --gradient_accumulation_steps 64 --max_sequence_length 2048 --seed 42 \
  --steer_x_factor 1.5 --fapm_k_ratio 0.1 > /dev/null
printf '{"weights":"fp32 master weights, bf16 autocast compute","optim":"%s","model_root":"%s","reused_from":"%s","reused_models":["base","lora","steer_lora"]}\n' \
  "$OPTIM" "$MODEL_ROOT" "$REFERENCE_ROOT" > "$REPORT_ROOT/precision_manifest.json"

shared=(--model_name_or_path "$BASE_MODEL" --train_data_path data/samsum_emnlp2019/train
        --dev_data_path data/samsum_emnlp2019/dev --per_device_train_batch_size 1
        --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 --learning_rate 1e-5
        --num_train_epochs 2 --max_seq_length 2048 --seed 42 --optim "$OPTIM")

if ! done_model "$MODEL_ROOT/full_sft"; then
  phase train_full_sft
  "$PYTHON_BIN" -u methods/train_sft.py "${shared[@]}" --model_dtype float32 \
    --output_dir "$MODEL_ROOT/full_sft" > "$LOG_DIR/train_full_sft.log" 2>&1
fi
if ! done_model "$MODEL_ROOT/steer_full"; then
  phase train_steer_full
  "$PYTHON_BIN" -u methods/train_steer.py "${shared[@]}" --x_factor 1.5 --disable_peft \
    --full_ft_dtype float32 --output_dir "$MODEL_ROOT/steer_full" > "$LOG_DIR/train_steer_full.log" 2>&1
fi
if ! done_model "$MODEL_ROOT/fapm"; then
  phase fapm
  "$PYTHON_BIN" -u methods/fapm.py --base_model "$BASE_MODEL" --sft_model "$MODEL_ROOT/full_sft" \
    --output_dir "$MODEL_ROOT/fapm" --k_ratio 0.1 > "$LOG_DIR/fapm.log" 2>&1
fi

for name in full_sft steer_full fapm; do
  if [[ ! -f "$REPORT_ROOT/$name/downstream_results.json" ]]; then
    phase "samsum_$name"
    "$PYTHON_BIN" -u eval/eval_samsum.py --model_path "$MODEL_ROOT/$name" \
      --output_dir "$REPORT_ROOT/$name" > "$LOG_DIR/samsum_$name.log" 2>&1
  fi
done

# Same probe as the reference margin-mechanism run: every SAMSum dev example and token.
paths=(--model-path "full_sft=$MODEL_ROOT/full_sft" --model-path "steer_full=$MODEL_ROOT/steer_full"
       --model-path "fapm=$MODEL_ROOT/fapm" --model-path "lora=$REFERENCE_ROOT/models/lora"
       --model-path "steer_lora=$REFERENCE_ROOT/models/steer_lora")
if [[ ! -f "$REPORT_ROOT/margin-mechanism/samsum/summary.json" ]]; then
  phase distribution_samsum
  "$PYTHON_BIN" -u eval/eval_kl.py --base-model "$BASE_MODEL" \
    --output-dir "$REPORT_ROOT/margin-mechanism/samsum" --samples 100000 --tokens-per-sample 100000 \
    --margin-diagnostics "${paths[@]}" > "$LOG_DIR/distribution_samsum.log" 2>&1
fi
for prompt_format in raw chat; do
  directory="$REPORT_ROOT/margin-mechanism/boolq$([[ $prompt_format == chat ]] && echo -chat || true)"
  if [[ ! -f "$directory/summary.json" ]]; then
    phase "boolq_$prompt_format"
    "$PYTHON_BIN" -u eval/analyze_boolq_margin.py --base-model "$BASE_MODEL" \
      --documents "$REFERENCE_ROOT/reports/margin-mechanism/boolq.validation.jsonl" \
      --prompt-format "$prompt_format" --output-dir "$directory" "${paths[@]}" \
      > "$LOG_DIR/boolq_$prompt_format.log" 2>&1
  fi
done

# Five capability benchmarks for the three new models; existing results are validated and reused.
phase benchmarks
"$PYTHON_BIN" -u benchmarks/stable_five/run.py --output-dir "$ARTIFACT_ROOT/stable-five" \
  --model-path "full_sft=$MODEL_ROOT/full_sft" --model-path "steer_full=$MODEL_ROOT/steer_full" \
  --model-path "fapm=$MODEL_ROOT/fapm" --models full_sft steer_full fapm > "$LOG_DIR/benchmarks.log" 2>&1
phase complete
