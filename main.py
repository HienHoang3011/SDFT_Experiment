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
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--sft_model", type=str, default="outputs/sft-qwen2-7b")
    parser.add_argument("--sft_lora_model", type=str, default="outputs/sft-lora-qwen2-7b")
    parser.add_argument("--fapm_model", type=str, default="outputs/fapm-qwen2-7b")
    parser.add_argument("--steer_model", type=str, default="outputs/steer-qwen2-7b")
    parser.add_argument("--steer_full_model", type=str, default="outputs/steer-full-qwen2-7b")
    
    parser.add_argument("--skip_train_sft", action="store_true")
    parser.add_argument("--skip_train_sft_lora", action="store_true")
    parser.add_argument("--skip_apply_fapm", action="store_true")
    parser.add_argument("--skip_train_steer", action="store_true")
    parser.add_argument("--skip_train_steer_full", action="store_true")
    
    parser.add_argument("--skip_eval_base", action="store_true")
    parser.add_argument("--skip_eval_sft", action="store_true")
    parser.add_argument("--skip_eval_sft_lora", action="store_true")
    parser.add_argument("--skip_eval_fapm", action="store_true")
    parser.add_argument("--skip_eval_steer", action="store_true")
    parser.add_argument("--skip_eval_steer_full", action="store_true")
    
    parser.add_argument("--skip_report", action="store_true")
    parser.add_argument("--disable_peft", action="store_true")
    
    args = parser.parse_args()

    eval_dirs = {
        "base_medqa": "outputs/base_medqa_eval",
        "base_prior": "outputs/base_prior_eval",
        "sft_medqa": "outputs/sft_medqa_eval",
        "sft_prior": "outputs/sft_prior_eval",
        "sft_lora_medqa": "outputs/sft_lora_medqa_eval",
        "sft_lora_prior": "outputs/sft_lora_prior_eval",
        "fapm_medqa": "outputs/fapm_medqa_eval",
        "fapm_prior": "outputs/fapm_prior_eval",
        "steer_medqa": "outputs/steer_medqa_eval",
        "steer_prior": "outputs/steer_prior_eval",
        "steer_full_medqa": "outputs/steer_full_medqa_eval",
        "steer_full_prior": "outputs/steer_full_prior_eval",
    }
    
    for d in eval_dirs.values():
        os.makedirs(d, exist_ok=True)
        
    peft_flag = " --disable_peft" if args.disable_peft else ""

    # ====================================================================
    # HUẤN LUYỆN MODEL
    # ====================================================================

    if not args.skip_train_sft:
        train_sft_cmd = f"/root/.local/bin/uv run python methods/train_sft.py --model_name_or_path {args.base_model} --output_dir {args.sft_model}{peft_flag}"
        run_command(train_sft_cmd, "BƯỚC 1A: HUẤN LUYỆN FULL SFT")
    else:
        print("\n⏭️  Bỏ qua Bước 1A: Train Full SFT")
        
    if not args.skip_train_sft_lora:
        train_sft_lora_cmd = f"/root/.local/bin/uv run python methods/train_sft_lora.py --model_name_or_path {args.base_model} --output_dir {args.sft_lora_model}"
        run_command(train_sft_lora_cmd, "BƯỚC 1B: HUẤN LUYỆN SFT LORA")
    else:
        print("\n⏭️  Bỏ qua Bước 1B: Train SFT LoRA")

    if not args.skip_apply_fapm:
        apply_fapm_cmd = f"/root/.local/bin/uv run python methods/fapm.py --base_model {args.base_model} --sft_model {args.sft_model} --output_dir {args.fapm_model}"
        run_command(apply_fapm_cmd, "BƯỚC 1C: ÁP DỤNG FAPM (MERGE SFT VÀO BASE)")
    else:
        print("\n⏭️  Bỏ qua Bước 1C: Apply FAPM")

    if not args.skip_train_steer:
        train_steer_cmd = f"/root/.local/bin/uv run python methods/train_steer.py --model_name_or_path {args.base_model} --output_dir {args.steer_model}{peft_flag}"
        run_command(train_steer_cmd, "BƯỚC 1D: HUẤN LUYỆN STEERED SFT")
    else:
        print("\n⏭️  Bỏ qua Bước 1D: Train Steered SFT")

    if not args.skip_train_steer_full:
        train_steer_full_cmd = f"/root/.local/bin/uv run python methods/train_steer.py --model_name_or_path {args.base_model} --output_dir {args.steer_full_model} --disable_peft"
        run_command(train_steer_full_cmd, "BƯỚC 1E: HUẤN LUYỆN STEERED SFT FULL")
    else:
        print("\n⏭️  Bỏ qua Bước 1E: Train Steered SFT Full")

    # ====================================================================
    # ĐÁNH GIÁ BASE MODEL
    # ====================================================================

    if not args.skip_eval_base:
        run_command(f"/root/.local/bin/uv run python eval/eval_medqa.py --model_path {args.base_model} --output_dir {eval_dirs['base_medqa']}", "BƯỚC 2A: ĐÁNH GIÁ MEDQA CHO BASE MODEL")
        run_command(f"/root/.local/bin/uv run python eval/eval_previous_task.py --model_path {args.base_model} --output_dir {eval_dirs['base_prior']}", "BƯỚC 2B: ĐÁNH GIÁ PRIOR CHO BASE MODEL")
    else:
        print("\n⏭️  Bỏ qua Bước 2: Đánh giá Base Model")

    # ====================================================================
    # ĐÁNH GIÁ SFT MODEL
    # ====================================================================

    if not args.skip_eval_sft:
        run_command(f"/root/.local/bin/uv run python eval/eval_medqa.py --model_path {args.sft_model} --output_dir {eval_dirs['sft_medqa']}", "BƯỚC 3A: ĐÁNH GIÁ MEDQA CHO FULL SFT")
        run_command(f"/root/.local/bin/uv run python eval/eval_previous_task.py --model_path {args.sft_model} --output_dir {eval_dirs['sft_prior']}", "BƯỚC 3B: ĐÁNH GIÁ PRIOR CHO FULL SFT")
    else:
        print("\n⏭️  Bỏ qua Bước 3: Đánh giá Full SFT Model")
        
    # ====================================================================
    # ĐÁNH GIÁ SFT LORA MODEL
    # ====================================================================

    if not args.skip_eval_sft_lora:
        run_command(f"/root/.local/bin/uv run python eval/eval_medqa.py --model_path {args.sft_lora_model} --output_dir {eval_dirs['sft_lora_medqa']}", "BƯỚC 4A: ĐÁNH GIÁ MEDQA CHO SFT LORA")
        run_command(f"/root/.local/bin/uv run python eval/eval_previous_task.py --model_path {args.sft_lora_model} --output_dir {eval_dirs['sft_lora_prior']}", "BƯỚC 4B: ĐÁNH GIÁ PRIOR CHO SFT LORA")
    else:
        print("\n⏭️  Bỏ qua Bước 4: Đánh giá SFT LoRA Model")

    # ====================================================================
    # ĐÁNH GIÁ FAPM MODEL
    # ====================================================================

    if not args.skip_eval_fapm:
        run_command(f"/root/.local/bin/uv run python eval/eval_medqa.py --model_path {args.fapm_model} --output_dir {eval_dirs['fapm_medqa']}", "BƯỚC 5A: ĐÁNH GIÁ MEDQA CHO FAPM")
        run_command(f"/root/.local/bin/uv run python eval/eval_previous_task.py --model_path {args.fapm_model} --output_dir {eval_dirs['fapm_prior']}", "BƯỚC 5B: ĐÁNH GIÁ PRIOR CHO FAPM")
    else:
        print("\n⏭️  Bỏ qua Bước 5: Đánh giá FAPM Model")

    # ====================================================================
    # ĐÁNH GIÁ STEERED SFT MODEL
    # ====================================================================

    if not args.skip_eval_steer:
        run_command(f"/root/.local/bin/uv run python eval/eval_medqa.py --model_path {args.steer_model} --output_dir {eval_dirs['steer_medqa']}", "BƯỚC 6A: ĐÁNH GIÁ MEDQA CHO STEERED MODEL")
        run_command(f"/root/.local/bin/uv run python eval/eval_previous_task.py --model_path {args.steer_model} --output_dir {eval_dirs['steer_prior']}", "BƯỚC 6B: ĐÁNH GIÁ PRIOR CHO STEERED MODEL")
    else:
        print("\n⏭️  Bỏ qua Bước 6: Đánh giá Steered SFT Model")

    if not args.skip_eval_steer_full:
        run_command(f"/root/.local/bin/uv run python eval/eval_medqa.py --model_path {args.steer_full_model} --output_dir {eval_dirs['steer_full_medqa']}", "BƯỚC 6C: ĐÁNH GIÁ MEDQA CHO STEERED FULL MODEL")
        run_command(f"/root/.local/bin/uv run python eval/eval_previous_task.py --model_path {args.steer_full_model} --output_dir {eval_dirs['steer_full_prior']}", "BƯỚC 6D: ĐÁNH GIÁ PRIOR CHO STEERED FULL MODEL")
    else:
        print("\n⏭️  Bỏ qua Bước 6 (phần Full): Đánh giá Steered SFT Full Model")

    # ====================================================================
    # SINH BÁO CÁO VÀ BIỂU ĐỒ
    # ====================================================================
    
    if not args.skip_report:
        print("\n" + "="*80)
        print("🚀 BƯỚC 7: TỔNG HỢP VÀ VẼ BIỂU ĐỒ BÁO CÁO")
        print("="*80 + "\n")
        
        from eval.generate_report import generate_report

        generate_report(
            base_new_task=f"{eval_dirs['base_medqa']}/eval_results.json",
            base_prior_task=f"{eval_dirs['base_prior']}/previous_capabilities_summary.json",
            sft_new_task=f"{eval_dirs['sft_medqa']}/eval_results.json",
            sft_prior_task=f"{eval_dirs['sft_prior']}/previous_capabilities_summary.json",
            sft_lora_new_task=f"{eval_dirs['sft_lora_medqa']}/eval_results.json",
            sft_lora_prior_task=f"{eval_dirs['sft_lora_prior']}/previous_capabilities_summary.json",
            fapm_new_task=f"{eval_dirs['fapm_medqa']}/eval_results.json",
            fapm_prior_task=f"{eval_dirs['fapm_prior']}/previous_capabilities_summary.json",
            steer_new_task=f"{eval_dirs['steer_medqa']}/eval_results.json",
            steer_prior_task=f"{eval_dirs['steer_prior']}/previous_capabilities_summary.json",
            steer_full_new_task=f"{eval_dirs['steer_full_medqa']}/eval_results.json",
            steer_full_prior_task=f"{eval_dirs['steer_full_prior']}/previous_capabilities_summary.json",
            output_dir="reports"
        )
    else:
        print("\n⏭️  Bỏ qua Bước 7: Sinh báo cáo")
        
    print("\n✅ PIPELINE CHẠY THÀNH CÔNG TỪ ĐẦU ĐẾN CUỐI (HOẶC ĐÃ BỎ QUA CÁC BƯỚC ĐƯỢC CHỈ ĐỊNH)!")

if __name__ == "__main__":
    main()
