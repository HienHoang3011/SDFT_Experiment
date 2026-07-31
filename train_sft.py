import argparse
import os
from datasets import Dataset, DatasetDict
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT Training with LoRA")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Path to the model or HF model name")
    parser.add_argument("--train_data_path", type=str, default="data/science_data/train_data",
                        help="Path to the train dataset (HF disk format)")
    parser.add_argument("--eval_data_path", type=str, default="data/science_data/eval_data",
                        help="Path to the eval dataset (HF disk format)")
    parser.add_argument("--output_dir", type=str, default="outputs/sft-qwen2.5-7b",
                        help="Directory to save the trained model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                        help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=1536, # max_prompt_length (1024) + max_completion_length (512)
                        help="Maximum sequence length")
    parser.add_argument("--disable_peft", action="store_true", 
                        help="Vô hiệu hóa LoRA (Full Fine-Tuning)")
    return parser.parse_args()

def prepare_train_dataset(dataset_path, model_name_or_path):
    print(f"Loading train dataset from {dataset_path}")
    dataset = Dataset.load_from_disk(dataset_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if "messages" in dataset.column_names:
        dataset = dataset.rename_column("messages", "prompt")
        
    def format_text(example):
        msgs = example["prompt"]
        if isinstance(msgs, list):
            # Append the assistant's answer to the conversation
            msgs_copy = list(msgs)
            msgs_copy.append({"role": "assistant", "content": example["output_text"]})
            example["text"] = tokenizer.apply_chat_template(
                msgs_copy, 
                tokenize=False, 
                add_generation_prompt=False
            )
        else:
            example["text"] = str(msgs) + "\n" + str(example["output_text"])
        return example
        
    dataset = dataset.map(format_text, desc="Formatting conversation text")
        
    return dataset

def main():
    args = parse_args()
    
    full_train_dataset = prepare_train_dataset(args.train_data_path, args.model_name_or_path)
    
    total_examples = len(full_train_dataset)
    print(f"\n[INFO] Tổng số lượng ví dụ trong tập train_data là: {total_examples}")
    
    print("[INFO] Đang chia (split) tập train thành 95% train / 5% eval...")
    split_dataset = full_train_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f" - Số lượng train: {len(train_dataset)}")
    print(f" - Số lượng eval: {len(eval_dataset)}")
    
    print("\n[Kiểm tra Dữ liệu Train mẫu]")
    print(train_dataset[0]["text"][:500] + "...")
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
        
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="steps",
        save_steps=0.2,
        report_to="none",
        bf16=True,
        fp16=False
    )
    
    peft_config = None
    if not args.disable_peft:
        peft_config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
    
    print(f"\nKhởi tạo SFTTrainer với model: {args.model_name_or_path}")
    
    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    print("\nBắt đầu quá trình huấn luyện SFT...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    print(f"Huấn luyện hoàn tất! Model lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()
