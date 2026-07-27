#!/bin/bash
set -e

# Cấu hình tham số (có thể thay đổi nếu cần)
BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_MODEL_DIR="outputs/sdft-qwen2.5-7b"

# Tạo các thư mục lưu kết quả
mkdir -p outputs/base_science_eval
mkdir -p outputs/base_prior_eval
mkdir -p outputs/sdft_science_eval
mkdir -p outputs/sdft_prior_eval
mkdir -p reports

echo "================================================================================"
echo "🚀 BƯỚC 2A: ĐÁNH GIÁ SCIENCE TASK CHO BASE MODEL"
echo "================================================================================"
# (Ghi chú: Bước này bạn đã chạy thành công trước đó, bạn có thể comment lại dòng dưới nếu muốn bỏ qua)
# uv run python eval/eval_science.py --model_path $BASE_MODEL --output_dir outputs/base_science_eval

echo "================================================================================"
echo "🚀 BƯỚC 2B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO BASE MODEL"
echo "================================================================================"
uv run python eval/eval_previous_task.py --model_path $BASE_MODEL --backend hf --output_dir outputs/base_prior_eval

echo "================================================================================"
echo "🚀 BƯỚC 3A: ĐÁNH GIÁ SCIENCE TASK CHO SDFT MODEL"
echo "================================================================================"
uv run python eval/eval_science.py --model_path $OUTPUT_MODEL_DIR --output_dir outputs/sdft_science_eval

echo "================================================================================"
echo "🚀 BƯỚC 3B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO SDFT MODEL"
echo "================================================================================"
uv run python eval/eval_previous_task.py --model_path $OUTPUT_MODEL_DIR --backend hf --output_dir outputs/sdft_prior_eval

echo "================================================================================"
echo "🚀 BƯỚC 4: TỔNG HỢP VÀ VẼ BIỂU ĐỒ BÁO CÁO"
echo "================================================================================"
uv run python -c "
from eval.generate_report import generate_report
generate_report(
    base_new_task='outputs/base_science_eval/eval_results.json',
    base_prior_task='outputs/base_prior_eval/previous_capabilities_summary.json',
    sdft_new_task='outputs/sdft_science_eval/eval_results.json',
    sdft_prior_task='outputs/sdft_prior_eval/previous_capabilities_summary.json',
    output_dir='reports'
)
"

echo "✅ PIPELINE ĐÁNH GIÁ ĐÃ HOÀN TẤT!"
