import argparse
import os
import json
import logging

try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("Vui lòng cài đặt thư viện lm-evaluation-harness để chạy file này:")
    print("pip install lm-eval")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate model on previous capabilities benchmarks")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--backend", type=str, default="vllm", choices=["vllm", "hf"],
                        help="Backend to use for evaluation (vllm or huggingface)")
    parser.add_argument("--batch_size", type=str, default="auto",
                        help="Batch size cho evaluation")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Thư mục lưu kết quả (mặc định lưu vào thư mục model)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Danh sách các benchmark đo lường "Previous Capabilities"
    # - IFEval và HumanEval cũng có trong lm_eval, nhưng HumanEval cần môi trường cho phép thực thi code.
    tasks = [
        "hellaswag",
        "mmlu",
        "truthfulqa",
        "humaneval",
        "ifeval" 
    ]
    
    print(f"Bắt đầu đánh giá mô hình {args.model_path} trên các task:")
    print(", ".join(tasks))
    
    # Khởi tạo HFLM trực tiếp để có thể vô hiệu hóa max_new_tokens=2048
    # được đóng gói sẵn trong generation_config của Qwen2 base. Nếu giữ nó,
    # Transformers sẽ bỏ qua giới hạn max_length/max_gen_toks của từng task.
    is_peft = os.path.exists(os.path.join(args.model_path, "adapter_config.json"))
    if is_peft:
        with open(os.path.join(args.model_path, "adapter_config.json"), "r") as f:
            config = json.load(f)
        base_model = config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("adapter_config.json is missing base_model_name_or_path")
        tokenizer_source = (
            args.model_path
            if os.path.exists(os.path.join(args.model_path, "tokenizer_config.json"))
            else base_model
        )
        eval_model = HFLM(
            pretrained=base_model,
            peft=args.model_path,
            tokenizer=tokenizer_source,
            dtype="auto",
            parallelize=True,
            batch_size=args.batch_size,
        )
    else:
        eval_model = HFLM(
            pretrained=args.model_path,
            tokenizer=args.model_path,
            dtype="auto",
            parallelize=True,
            batch_size=args.batch_size,
        )

    generation_config = eval_model.model.generation_config
    inherited_max_new_tokens = generation_config.max_new_tokens
    generation_config.max_new_tokens = None
    generation_config.eos_token_id = eval_model.tokenizer.eos_token_id
    generation_config.pad_token_id = eval_model.tokenizer.pad_token_id
    eval_model.model.config.eos_token_id = eval_model.tokenizer.eos_token_id
    eval_model.model.config.pad_token_id = eval_model.tokenizer.pad_token_id
    print(
        "[Generation config] Disabled inherited max_new_tokens="
        f"{inherited_max_new_tokens}; lm-eval task limits will be used."
    )
    print(
        "[Generation config] "
        f"EOS={eval_model.tokenizer.eos_token} "
        f"({eval_model.tokenizer.eos_token_id}), "
        f"PAD={eval_model.tokenizer.pad_token} "
        f"({eval_model.tokenizer.pad_token_id})."
    )

    # Để đánh giá HumanEval bằng thư viện, đôi khi cần thiết lập biến môi trường
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Chạy đánh giá
    results = lm_eval.simple_evaluate(
        model=eval_model,
        tasks=tasks,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )
    
    # Xử lý thư mục lưu kết quả
    output_dir = args.output_dir if args.output_dir else args.model_path
    os.makedirs(output_dir, exist_ok=True)
    
    metrics = {}
    print("\n" + "=" * 60)
    print("KẾT QUẢ ĐÁNH GIÁ (PREVIOUS CAPABILITIES):")
    
    if results is not None and "results" in results:
        task_results = results["results"]
        total_score = 0.0
        valid_tasks = 0
        
        # Hàm phụ để lấy ra metric phù hợp nhất của từng benchmark
        def get_primary_metric(res_dict):
            """
            Tìm metric chính xác cho từng loại task.
            Ưu tiên acc_norm, sau đó acc, exact_match, pass@1, bleu_acc, rouge1_acc.
            """
            primary_metrics = [
                'acc_norm,none', 
                'acc,none', 
                'exact_match,none', 
                'prompt_level_strict_acc,none', 
                'pass@1,create_test',
                'pass@1,none',
                'rouge1_acc,none',
                'bleu_acc,none'
            ]
            for metric_name in primary_metrics:
                if metric_name in res_dict:
                    return res_dict[metric_name]
            
            # Nếu không thấy, lấy bừa metric số đầu tiên
            for k, v in res_dict.items():
                if isinstance(v, (int, float)) and 'stderr' not in k:
                    return v
            return 0.0

        for task_name, res in task_results.items():
            score = get_primary_metric(res)
            metrics[task_name] = score
            print(f"  {task_name}: {score:.4f}")
            if 0 <= score <= 1:
                total_score += score
                valid_tasks += 1
            
        if valid_tasks > 0:
            avg_score = total_score / valid_tasks
            metrics["average_score"] = avg_score
            print("-" * 60)
            print(f"  TRUNG BÌNH (Catastrophic Forgetting Measure): {avg_score:.4f}")
    else:
        print("Không có kết quả trả về từ lm_eval.")
        
    print("=" * 60)
    
    # Lưu kết quả chi tiết
    output_path = os.path.join(output_dir, "previous_capabilities_detailed.json")
    with open(output_path, "w", encoding="utf-8") as f:
        # Cần extract phần dict để json dump không bị lỗi với các obj của thư viện
        json.dump(results["results"] if results else {}, f, indent=2, ensure_ascii=False)
        
    # Lưu điểm số tổng hợp
    summary_path = os.path.join(output_dir, "previous_capabilities_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
        
    print(f"\nĐã lưu kết quả chi tiết tại: {output_path}")
    print(f"Đã lưu kết quả tóm tắt tại: {summary_path}")

if __name__ == "__main__":
    main()
