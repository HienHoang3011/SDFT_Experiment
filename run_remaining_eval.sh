#!/bin/bash
set -e

echo "================================================================================"
echo "🚀 BƯỚC 2B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO BASE MODEL"
echo "================================================================================"
uv run python eval/eval_previous_task.py --model_path Qwen/Qwen2.5-7B-Instruct --output_dir outputs/base_prior_eval

echo "================================================================================"
echo "🚀 BƯỚC 3A: ĐÁNH GIÁ SCIENCE TASK CHO SDFT MODEL (MERGED)"
echo "================================================================================"
uv run python eval/eval_science.py --model_path outputs/sdft-qwen2.5-7b-merged --output_dir outputs/sdft_science_eval

echo "================================================================================"
echo "🚀 BƯỚC 3B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO SDFT MODEL (MERGED)"
echo "================================================================================"
uv run python eval/eval_previous_task.py --model_path outputs/sdft-qwen2.5-7b-merged --output_dir outputs/sdft_prior_eval

echo "================================================================================"
echo "🚀 BƯỚC 4A: ĐÁNH GIÁ SCIENCE TASK CHO SFT MODEL (MERGED)"
echo "================================================================================"
uv run python eval/eval_science.py --model_path outputs/sft-qwen2.5-7b-merged --output_dir outputs/sft_science_eval

echo "================================================================================"
echo "🚀 BƯỚC 4B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO SFT MODEL (MERGED)"
echo "================================================================================"
uv run python eval/eval_previous_task.py --model_path outputs/sft-qwen2.5-7b-merged --output_dir outputs/sft_prior_eval

echo "================================================================================"
echo "🚀 BƯỚC 5A: ĐÁNH GIÁ SCIENCE TASK CHO STEERED SFT MODEL (MERGED)"
echo "================================================================================"
uv run python eval/eval_science.py --model_path outputs/steer-qwen2.5-7b-merged --output_dir outputs/steer_science_eval

echo "================================================================================"
echo "🚀 BƯỚC 5B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO STEERED SFT MODEL (MERGED)"
echo "================================================================================"
uv run python eval/eval_previous_task.py --model_path outputs/steer-qwen2.5-7b-merged --output_dir outputs/steer_prior_eval

echo "================================================================================"
echo "🚀 BƯỚC 6: TỔNG HỢP VÀ VẼ BIỂU ĐỒ BÁO CÁO"
echo "================================================================================"
uv run python -c "
from eval.generate_report import generate_report; 
generate_report(
    base_new_task='outputs/base_science_eval/eval_results.json', 
    base_prior_task='outputs/base_prior_eval/previous_capabilities_summary.json', 
    sdft_new_task='outputs/sdft_science_eval/eval_results.json', 
    sdft_prior_task='outputs/sdft_prior_eval/previous_capabilities_summary.json', 
    sft_new_task='outputs/sft_science_eval/eval_results.json', 
    sft_prior_task='outputs/sft_prior_eval/previous_capabilities_summary.json', 
    steer_new_task='outputs/steer_science_eval/eval_results.json', 
    steer_prior_task='outputs/steer_prior_eval/previous_capabilities_summary.json', 
    output_dir='reports'
)"

echo "✅ HOÀN TẤT TOÀN BỘ PIPELINE ĐÁNH GIÁ!"
