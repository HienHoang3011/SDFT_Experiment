import argparse
import os
from datasets import Dataset, DatasetDict
from trl.experimental.sdft import SDFTConfig, SDFTTrainer
from peft import LoraConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run SDFT Training with LoRA")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Path to the model or HF model name")
    parser.add_argument("--train_data_path", type=str, default="data/science_data/train_data",
                        help="Path to the train dataset (HF disk format)")
    parser.add_argument("--eval_data_path", type=str, default="data/science_data/eval_data",
                        help="Path to the eval dataset (HF disk format)")
    parser.add_argument("--output_dir", type=str, default="outputs/sdft-qwen2.5-7b",
                        help="Directory to save the trained model")
    parser.add_argument("--teacher_model_kind", type=str, default="base", choices=["base", "live", "ema"],
                        help="Kind of teacher model to use (default: base)")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_prompt_length", type=int, default=1024,
                        help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Maximum completion length")
    parser.add_argument("--use_vllm", action="store_true",
                        help="Enable vLLM generation acceleration if installed")
    return parser.parse_args()

def prepare_train_dataset(dataset_path, model_name_or_path):
    print(f"Loading train dataset from {dataset_path}")
    dataset = Dataset.load_from_disk(dataset_path)
    from transformers import AutoTokenizer
    import re
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # Train data thực tế của bạn có các cột: 'messages', 'output_text'
    # SDFT yêu cầu bắt buộc: 'prompt', 'privileged_context'
    if "messages" in dataset.column_names:
        dataset = dataset.rename_column("messages", "prompt")
        
    def create_privileged_context(example):
        match = re.search(r'<answer>\s*(.*?)\s*</answer>', example['output_text'], flags=re.DOTALL)
        ans = match.group(1).strip() if match else "unknown"
        example["privileged_context"] = f"Hint: The correct answer is {ans}."
        
        if isinstance(example["prompt"], list):
            example["prompt"] = tokenizer.apply_chat_template(
                example["prompt"], 
                tokenize=False, 
                add_generation_prompt=True
            )
        return example
        
    dataset = dataset.map(create_privileged_context, desc="Creating hints and formatting prompts")
        
    return dataset

def prepare_eval_dataset(dataset_path):
    print(f"Loading eval dataset from {dataset_path}")
    dataset = Dataset.load_from_disk(dataset_path)
    
    # Eval data thực tế của bạn có các cột: 'prompt', 'answer'
    if "answer" in dataset.column_names:
        dataset = dataset.rename_column("answer", "privileged_context")
        
    return dataset

def main():
    args = parse_args()
    
    # 1. Chuẩn bị dữ liệu dựa theo schema thực tế trong file .arrow
    # 2. Chuẩn bị dataset
    full_train_dataset = prepare_train_dataset(args.train_data_path, args.model_name_or_path)
    
    # Kích thước tập dữ liệu
    total_examples = len(full_train_dataset)
    print(f"\n[INFO] Tổng số lượng ví dụ trong tập train_data là: {total_examples}")
    
    # Chia data 80/20 theo yêu cầu để lấy tập valid
    print("[INFO] Đang chia (split) tập train thành 95% train / 5% eval...")
    split_dataset = full_train_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f" - Số lượng train: {len(train_dataset)}")
    print(f" - Số lượng eval: {len(eval_dataset)}")
    
    print("\n[Kiểm tra Dữ liệu Train mẫu]")
    print(train_dataset[0])
    
    # 2. Cấu hình SDFT (Giống chuẩn trong docs của TRL)
    training_args = SDFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        
        # Cấu hình Teacher như docs (teacher_model_kind="base" tương đương với PEFT adapter bị disable lúc lấy teacher forward)
        teacher_model_kind=args.teacher_model_kind,
        # teacher_prompt_template=(
        #     "{prompt}\n\n"
        #     "This is an example for a response to the question:\n"
        #     "{privileged_context}\n\n"
        #     "Now answer with a response of your own, including the thinking process."
        # ),
        
        # Cấu hình Distillation (mặc định như docs)
        distillation_alpha=0.5,
        distillation_mode="topk_logits",
        distillation_topk=5,
        
        # Tùy chọn vLLM nếu truyền cờ
        use_vllm=args.use_vllm,
        
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="steps",
        save_steps=0.2,
        report_to="none",
        bf16=True,
        fp16=False

    )
    
    # 3. Cấu hình LoRA (PEFT)
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    
    # 4. Khởi tạo SDFTTrainer với LoRA
    print(f"\nKhởi tạo SDFTTrainer với model: {args.model_name_or_path}")
    print(f"Teacher model kind: {args.teacher_model_kind}")
    
    trainer = SDFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config, # Truyền LoRA vào SDFTTrainer
    )
    
    # Ép SDFTTrainer sinh bằng Greedy Decoding
    if hasattr(trainer, "generation_kwargs"):
        trainer.generation_kwargs["do_sample"] = False
        trainer.generation_kwargs.pop("temperature", None)
        trainer.generation_kwargs.pop("top_p", None)
        
    # 5. Chạy Training
    print("\nBắt đầu quá trình huấn luyện SDFT...")
    trainer.train()
    
    # Lưu model cuối cùng
    trainer.save_model(args.output_dir)
    print(f"Huấn luyện hoàn tất! Model lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()
