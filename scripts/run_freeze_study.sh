#!/usr/bin/env bash
# Freeze study: Full SFT fp32 with one block of 7 transformer layers frozen during training.
#   freeze_bottom  embeddings + layers 0-6     (Spurious Forgetting, ICLR'25)
#   freeze_middle  layers 14-20                (control: same number of frozen layers)
#   freeze_top     layers 21-27 + norm + head  (where the <answer> decision is read out)
# Same data, seed, LR and schedule as fp32-master-2epoch-20261012; its full_sft is the
# unfrozen reference. Every step is skipped when its output exists, so the script resumes.
#
# Usage: bash scripts/run_freeze_study.sh [arm ...]      (default: all three arms)
#        BENCH=1 bash scripts/run_freeze_study.sh        (also run the full 5 benchmarks, ~4 h/arm)
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-artifacts/base/Qwen2-7B}"
STUDY_ROOT="${STUDY_ROOT:-artifacts/samsum/freeze-study}"
MODEL_ROOT="$STUDY_ROOT/models"   # on /data: checkpoints must survive a pod restart
LOG_DIR="$STUDY_ROOT/logs"
mkdir -p "$MODEL_ROOT" "$LOG_DIR"
export PYTHONPATH="$ROOT_DIR/methods${PYTHONPATH:+:$PYTHONPATH}"

declare -A FREEZE=(
  [freeze_bottom]="--freeze_layers 0-6 --freeze_embeddings"
  [freeze_middle]="--freeze_layers 14-20"
  [freeze_top]="--freeze_layers 21-27 --freeze_head"
)
ARMS=("${@:-freeze_bottom freeze_middle freeze_top}")
read -r -a ARMS <<< "${ARMS[*]}"

phase() { printf '%s %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$1" | tee -a "$LOG_DIR/phases.log"; }
done_model() { [[ -f "$1/config.json" && -n "$(find "$1" -maxdepth 1 -name '*.safetensors' -print -quit)" ]]; }

if ! done_model "$BASE_MODEL"; then
  phase download_base
  "$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2-7B', local_dir='$BASE_MODEL')"
fi

for arm in "${ARMS[@]}"; do
  [[ -n "${FREEZE[$arm]:-}" ]] || { echo "Unknown arm $arm" >&2; exit 1; }
  model="$MODEL_ROOT/$arm"
  if ! done_model "$model"; then
    phase "train_$arm"
    # shellcheck disable=SC2086
    "$PYTHON_BIN" -u methods/train_sft.py --model_name_or_path "$BASE_MODEL" \
      --train_data_path data/samsum_emnlp2019/train --dev_data_path data/samsum_emnlp2019/dev \
      --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --gradient_accumulation_steps 64 \
      --learning_rate 1e-5 --num_train_epochs 2 --max_seq_length 2048 --seed 42 \
      --model_dtype float32 --optim paged_adamw_8bit --save_dtype bfloat16 ${FREEZE[$arm]} \
      --output_dir "$model" > "$LOG_DIR/train_$arm.log" 2>&1
  fi
  # Same probe as fapm-analysis/: ROUGE-L (819), GSM8K (250), BoolQ ChatML (1000),
  # IFEval stop/empty rates (200), KL to Base. Variant full_sft = the checkpoint itself.
  if [[ ! -f "$STUDY_ROOT/probe-$arm/probe_results.json" ]]; then
    phase "probe_$arm"
    "$PYTHON_BIN" -u eval/analyze_fapm_masks.py probe --base-model "$BASE_MODEL" --sft-model "$model" \
      --variants full_sft --ifeval-n 200 --output-dir "$STUDY_ROOT/probe-$arm" > "$LOG_DIR/probe_$arm.log" 2>&1
  fi
  if [[ "${BENCH:-0}" == 1 ]]; then
    if [[ ! -f "$STUDY_ROOT/reports/$arm/downstream_results.json" ]]; then
      phase "samsum_$arm"
      "$PYTHON_BIN" -u eval/eval_samsum.py --model_path "$model" \
        --output_dir "$STUDY_ROOT/reports/$arm" > "$LOG_DIR/samsum_$arm.log" 2>&1
    fi
    phase "benchmarks_$arm"
    "$PYTHON_BIN" -u benchmarks/stable_five/run.py --output-dir "$STUDY_ROOT/stable-five" \
      --model-path "$arm=$model" --models "$arm" > "$LOG_DIR/benchmarks_$arm.log" 2>&1
  fi
done
phase complete
