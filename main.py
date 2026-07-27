import argparse
import subprocess
import os
import sys

def run_command(command, description):
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print(f"Bắt đầu chạy: {command}")
    print("="*80 + "\n")
    
    # Dùng subprocess.run để thực thi lệnh trực tiếp
    # Việc này giúp tách biệt bộ nhớ CUDA cho từng bước
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n❌ LỖI: Bước '{description}' thất bại với mã lỗi {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="End-to-End SDFT Pipeline")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Tên hoặc đường dẫn model gốc (Base Model)")
    parser.add_argument("--output_model_dir", type=str, default="outputs/sdft-qwen2.5-7b",
                        help="Thư mục lưu model sau khi train SDFT")
    args = parser.parse_args()

    # Thư mục lưu kết quả đánh giá (eval)
    eval_dirs = {
        "base_science": "outputs/base_science_eval",
        "base_prior": "outputs/base_prior_eval",
        "sdft_science": "outputs/sdft_science_eval",
        "sdft_prior": "outputs/sdft_prior_eval",
    }
    
    for d in eval_dirs.values():
        os.makedirs(d, exist_ok=True)

    # 1. HUẤN LUYỆN MODEL VỚI SDFT
    # (Có thể chỉnh sửa batch_size hoặc số epoch trong lệnh này)
    train_cmd = f"python train_sdft.py --model_name_or_path {args.base_model} --output_dir {args.output_model_dir}"
    run_command(train_cmd, "BƯỚC 1: HUẤN LUYỆN MÔ HÌNH VỚI SDFT")

    # ====================================================================
    # ĐÁNH GIÁ BASE MODEL (MÔ HÌNH GỐC)
    # ====================================================================

    # 2. Đánh giá Science (Task Mới) cho Base Model
    eval_base_science_cmd = f"python eval/eval_science.py --model_path {args.base_model} --output_dir {eval_dirs['base_science']}"
    run_command(eval_base_science_cmd, "BƯỚC 2A: ĐÁNH GIÁ SCIENCE TASK CHO BASE MODEL")

    # 3. Đánh giá Previous Capabilities (Task Cũ) cho Base Model
    eval_base_prior_cmd = f"python eval/eval_previous_task.py --model_path {args.base_model} --output_dir {eval_dirs['base_prior']}"
    run_command(eval_base_prior_cmd, "BƯỚC 2B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO BASE MODEL")

    # ====================================================================
    # ĐÁNH GIÁ SDFT FINETUNED MODEL (MÔ HÌNH SAU KHI TRAIN)
    # ====================================================================

    # 4. Đánh giá Science cho SDFT Model
    eval_sdft_science_cmd = f"python eval/eval_science.py --model_path {args.output_model_dir} --output_dir {eval_dirs['sdft_science']}"
    run_command(eval_sdft_science_cmd, "BƯỚC 3A: ĐÁNH GIÁ SCIENCE TASK CHO SDFT MODEL")

    # 5. Đánh giá Previous Capabilities cho SDFT Model
    eval_sdft_prior_cmd = f"python eval/eval_previous_task.py --model_path {args.output_model_dir} --output_dir {eval_dirs['sdft_prior']}"
    run_command(eval_sdft_prior_cmd, "BƯỚC 3B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO SDFT MODEL")

    # ====================================================================
    # SINH BÁO CÁO VÀ BIỂU ĐỒ (GENERATE REPORT)
    # ====================================================================
    
    # 6. Gọi thẳng hàm generate_report từ script
    print("\n" + "="*80)
    print("🚀 BƯỚC 4: TỔNG HỢP VÀ VẼ BIỂU ĐỒ BÁO CÁO")
    print("="*80 + "\n")
    
    # Thêm sys.path để có thể import từ thư mục eval
    from eval.generate_report import generate_report

    generate_report(
        base_new_task=f"{eval_dirs['base_science']}/eval_results.json",
        base_prior_task=f"{eval_dirs['base_prior']}/previous_capabilities_summary.json",
        sdft_new_task=f"{eval_dirs['sdft_science']}/eval_results.json",
        sdft_prior_task=f"{eval_dirs['sdft_prior']}/previous_capabilities_summary.json",
        output_dir="reports"
    )
    
    print("\n✅ PIPELINE CHẠY THÀNH CÔNG TỪ ĐẦU ĐẾN CUỐI!")

if __name__ == "__main__":
    main()
