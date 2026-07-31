import argparse
import subprocess
import os
import sys

def run_command(command, description):
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print(f"Bắt đầu chạy: {command}")
    print("="*80 + "\n")
    
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"\n❌ LỖI: Bước '{description}' thất bại với mã lỗi {result.returncode}")
        sys.exit(result.returncode)

def main():
    parser = argparse.ArgumentParser(description="End-to-End SDFT, SFT and Steered SFT Pipeline")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Tên hoặc đường dẫn model gốc (Base Model)")
    parser.add_argument("--sdft_model", type=str, default="outputs/sdft-qwen2.5-7b",
                        help="Thư mục hoặc repo HF của model SDFT")
    parser.add_argument("--sft_model", type=str, default="outputs/sft-qwen2.5-7b",
                        help="Thư mục hoặc repo HF của model SFT")
    parser.add_argument("--steer_model", type=str, default="outputs/steer-qwen2.5-7b",
                        help="Thư mục hoặc repo HF của model Steered SFT")
    
    # Flags to skip phases
    parser.add_argument("--skip_train_sdft", action="store_true", help="Bỏ qua việc train SDFT")
    parser.add_argument("--skip_train_sft", action="store_true", help="Bỏ qua việc train SFT")
    parser.add_argument("--skip_train_steer", action="store_true", help="Bỏ qua việc train Steered SFT")
    
    parser.add_argument("--skip_eval_base", action="store_true", help="Bỏ qua đánh giá Base Model")
    parser.add_argument("--skip_eval_sdft", action="store_true", help="Bỏ qua đánh giá SDFT Model")
    parser.add_argument("--skip_eval_sft", action="store_true", help="Bỏ qua đánh giá SFT Model")
    parser.add_argument("--skip_eval_steer", action="store_true", help="Bỏ qua đánh giá Steered SFT Model")
    
    parser.add_argument("--skip_report", action="store_true", help="Bỏ qua việc sinh báo cáo")
    parser.add_argument("--disable_peft", action="store_true", help="Vô hiệu hóa LoRA (Full Fine-Tuning) cho tất cả các model khi train")
    
    args = parser.parse_args()

    eval_dirs = {
        "base_science": "outputs/base_science_eval",
        "base_prior": "outputs/base_prior_eval",
        "sdft_science": "outputs/sdft_science_eval",
        "sdft_prior": "outputs/sdft_prior_eval",
        "sft_science": "outputs/sft_science_eval",
        "sft_prior": "outputs/sft_prior_eval",
        "steer_science": "outputs/steer_science_eval",
        "steer_prior": "outputs/steer_prior_eval",
    }
    
    for d in eval_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    peft_flag = " --disable_peft" if args.disable_peft else ""

    # ====================================================================
    # HUẤN LUYỆN MODEL
    # ====================================================================

    if not args.skip_train_sdft:
        train_sdft_cmd = f"python train_sdft.py --model_name_or_path {args.base_model} --output_dir {args.sdft_model}{peft_flag}"
        run_command(train_sdft_cmd, "BƯỚC 1A: HUẤN LUYỆN MÔ HÌNH VỚI SDFT")
    else:
        print("\n⏭️  Bỏ qua Bước 1A: Train SDFT")

    if not args.skip_train_sft:
        train_sft_cmd = f"python train_sft.py --model_name_or_path {args.base_model} --output_dir {args.sft_model}{peft_flag}"
        run_command(train_sft_cmd, "BƯỚC 1B: HUẤN LUYỆN MÔ HÌNH VỚI SFT")
    else:
        print("\n⏭️  Bỏ qua Bước 1B: Train SFT")
        
    if not args.skip_train_steer:
        train_steer_cmd = f"python train_steer.py --model_name_or_path {args.base_model} --output_dir {args.steer_model}{peft_flag}"
        run_command(train_steer_cmd, "BƯỚC 1C: HUẤN LUYỆN MÔ HÌNH VỚI STEERED SFT")
    else:
        print("\n⏭️  Bỏ qua Bước 1C: Train Steered SFT")

    # ====================================================================
    # ĐÁNH GIÁ BASE MODEL
    # ====================================================================

    if not args.skip_eval_base:
        eval_base_science_cmd = f"python eval/eval_science.py --model_path {args.base_model} --output_dir {eval_dirs['base_science']}"
        run_command(eval_base_science_cmd, "BƯỚC 2A: ĐÁNH GIÁ SCIENCE TASK CHO BASE MODEL")

        eval_base_prior_cmd = f"python eval/eval_previous_task.py --model_path {args.base_model} --output_dir {eval_dirs['base_prior']}"
        run_command(eval_base_prior_cmd, "BƯỚC 2B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO BASE MODEL")
    else:
        print("\n⏭️  Bỏ qua Bước 2: Đánh giá Base Model")

    # ====================================================================
    # ĐÁNH GIÁ SDFT MODEL
    # ====================================================================

    if not args.skip_eval_sdft:
        eval_sdft_science_cmd = f"python eval/eval_science.py --model_path {args.sdft_model} --output_dir {eval_dirs['sdft_science']}"
        run_command(eval_sdft_science_cmd, "BƯỚC 3A: ĐÁNH GIÁ SCIENCE TASK CHO SDFT MODEL")

        eval_sdft_prior_cmd = f"python eval/eval_previous_task.py --model_path {args.sdft_model} --output_dir {eval_dirs['sdft_prior']}"
        run_command(eval_sdft_prior_cmd, "BƯỚC 3B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO SDFT MODEL")
    else:
        print("\n⏭️  Bỏ qua Bước 3: Đánh giá SDFT Model")

    # ====================================================================
    # ĐÁNH GIÁ SFT MODEL
    # ====================================================================

    if not args.skip_eval_sft:
        eval_sft_science_cmd = f"python eval/eval_science.py --model_path {args.sft_model} --output_dir {eval_dirs['sft_science']}"
        run_command(eval_sft_science_cmd, "BƯỚC 4A: ĐÁNH GIÁ SCIENCE TASK CHO SFT MODEL")

        eval_sft_prior_cmd = f"python eval/eval_previous_task.py --model_path {args.sft_model} --output_dir {eval_dirs['sft_prior']}"
        run_command(eval_sft_prior_cmd, "BƯỚC 4B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO SFT MODEL")
    else:
        print("\n⏭️  Bỏ qua Bước 4: Đánh giá SFT Model")
        
    # ====================================================================
    # ĐÁNH GIÁ STEERED SFT MODEL
    # ====================================================================

    if not args.skip_eval_steer:
        eval_steer_science_cmd = f"python eval/eval_science.py --model_path {args.steer_model} --output_dir {eval_dirs['steer_science']}"
        run_command(eval_steer_science_cmd, "BƯỚC 5A: ĐÁNH GIÁ SCIENCE TASK CHO STEERED MODEL")

        eval_steer_prior_cmd = f"python eval/eval_previous_task.py --model_path {args.steer_model} --output_dir {eval_dirs['steer_prior']}"
        run_command(eval_steer_prior_cmd, "BƯỚC 5B: ĐÁNH GIÁ PREVIOUS CAPABILITIES CHO STEERED MODEL")
    else:
        print("\n⏭️  Bỏ qua Bước 5: Đánh giá Steered SFT Model")

    # ====================================================================
    # SINH BÁO CÁO VÀ BIỂU ĐỒ
    # ====================================================================
    
    if not args.skip_report:
        print("\n" + "="*80)
        print("🚀 BƯỚC 6: TỔNG HỢP VÀ VẼ BIỂU ĐỒ BÁO CÁO")
        print("="*80 + "\n")
        
        # Thêm sys.path để có thể import từ thư mục eval
        from eval.generate_report import generate_report

        generate_report(
            base_new_task=f"{eval_dirs['base_science']}/eval_results.json",
            base_prior_task=f"{eval_dirs['base_prior']}/previous_capabilities_summary.json",
            sdft_new_task=f"{eval_dirs['sdft_science']}/eval_results.json",
            sdft_prior_task=f"{eval_dirs['sdft_prior']}/previous_capabilities_summary.json",
            sft_new_task=f"{eval_dirs['sft_science']}/eval_results.json",
            sft_prior_task=f"{eval_dirs['sft_prior']}/previous_capabilities_summary.json",
            steer_new_task=f"{eval_dirs['steer_science']}/eval_results.json",
            steer_prior_task=f"{eval_dirs['steer_prior']}/previous_capabilities_summary.json",
            output_dir="reports"
        )
    else:
        print("\n⏭️  Bỏ qua Bước 6: Sinh báo cáo")
        
    print("\n✅ PIPELINE CHẠY THÀNH CÔNG TỪ ĐẦU ĐẾN CUỐI (HOẶC ĐÃ BỎ QUA CÁC BƯỚC ĐƯỢC CHỈ ĐỊNH)!")

if __name__ == "__main__":
    main()
