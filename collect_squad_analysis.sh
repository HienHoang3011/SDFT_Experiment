#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT_DIR/evals_and_steer_images/squad_analysis}"
REPORT_DIR="${REPORT_DIR:-$ROOT_DIR/reports-squad}"
STEER_MODEL_DIR="${STEER_MODEL_DIR:-$ROOT_DIR/outputs/steer-full-squad-qwen2-7b}"

mkdir -p "$ANALYSIS_DIR/evals" "$ANALYSIS_DIR/report" \
  "$ANALYSIS_DIR/training_diagnostics" "$ANALYSIS_DIR/logs"

# Explicitly whitelist evaluation directories. Model/checkpoint directories are
# deliberately absent so this collector can never archive model weights.
eval_names=(
  base_squad_eval
  base_prior_eval
  sft_squad_eval
  sft_prior_eval
  sft_lora_squad_eval
  sft_lora_prior_eval
  steer_full_squad_eval
  steer_full_prior_eval
)

for eval_name in "${eval_names[@]}"; do
  source_dir="$ROOT_DIR/outputs/$eval_name"
  if [[ -d "$source_dir" ]]; then
    destination="$ANALYSIS_DIR/evals/$eval_name"
    mkdir -p "$destination"
    cp -a "$source_dir/." "$destination/"
  fi
done

if [[ -d "$REPORT_DIR" ]]; then
  cp -a "$REPORT_DIR/." "$ANALYSIS_DIR/report/"
fi

# The steer trainer writes analysis PNGs and scalar history beside its temporary
# checkpoint. Copy only diagnostics, never tokenizer/config/weight files.
for diagnostic_name in \
  base_finetune_drift.png \
  steering_margin_diagnostics.png \
  trainer_state.json; do
  diagnostic_path="$STEER_MODEL_DIR/$diagnostic_name"
  if [[ -f "$diagnostic_path" ]]; then
    cp -a "$diagnostic_path" "$ANALYSIS_DIR/training_diagnostics/"
  fi
done

for log_name in squad_eval_resume.log squad_steer_full.log; do
  log_path="$ROOT_DIR/logs/$log_name"
  if [[ -f "$log_path" ]]; then
    cp -a "$log_path" "$ANALYSIS_DIR/logs/"
  fi
done

(
  cd "$ANALYSIS_DIR"
  find . -type f ! -name MANIFEST.sha256 -print0 \
    | sort -z \
    | xargs -0 sha256sum
) > "$ANALYSIS_DIR/MANIFEST.sha256"

echo "Analysis artifacts collected in: $ANALYSIS_DIR"
