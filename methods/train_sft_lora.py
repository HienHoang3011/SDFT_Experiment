import argparse
import os
from datasets import Dataset, DatasetDict
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Run SFT Training with LoRA")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Path to the model or HF model name")
    parser.add_argument("--train_data_path", type=str, default="data/medqa_data/train_data",
                        help="Path to the train dataset (HF disk format)")
    parser.add_argument("--eval_data_path", type=str, default="data/medqa_data/eval_data",
                        help="Path to the eval dataset (HF disk format)")
    parser.add_argument("--dev_data_path", type=str, default="data/medqa_data/dev_data",
                        help="Path to the dev dataset for validation during training")
    parser.add_argument("--output_dir", type=str, default="outputs/sft-qwen2.5-7b-lora",
                        help="Directory to save the trained model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU for training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32,
                        help="Number of updates steps to accumulate before backward")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                        help="Total number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=2048, # max_prompt_length (1024) + max_completion_length (1024)
                        help="Maximum sequence length")
    return parser.parse_args()

def prepare_train_dataset(dataset_path, model_name_or_path):
    print(f"Loading train dataset from {dataset_path}")
    dataset = Dataset.load_from_disk(dataset_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def format_messages(example):
        msgs = example.get("messages", example.get("prompt", []))
        if isinstance(msgs, list):
            # Append the assistant's answer to the conversation
            msgs_copy = list(msgs)
            msgs_copy.append({"role": "assistant", "content": example["output_text"]})
            example["messages"] = msgs_copy
        else:
            example["messages"] = [{"role": "user", "content": str(msgs)}, {"role": "assistant", "content": str(example["output_text"])}]
        return example
        
    dataset = dataset.map(format_messages, desc="Formatting conversation messages")
    
    cols_to_remove = [col for col in ["prompt", "text", "output_text"] if col in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
        
    return dataset

def main():
    args = parse_args()
    
    train_dataset = prepare_train_dataset(args.train_data_path, args.model_name_or_path)
    eval_dataset = prepare_train_dataset(args.dev_data_path, args.model_name_or_path)
    
    print(f"\n[INFO] Số lượng tập train: {len(train_dataset)}")
    print(f"[INFO] Số lượng tập dev (validation): {len(eval_dataset)}")
    
    print("\n[Kiểm tra Dữ liệu Train mẫu]")
    print(str(train_dataset[0]["messages"])[:500] + "...")
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
        
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        report_to="none",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True
    )
    
    peft_config = LoraConfig(
        r=128,
        lora_alpha=256,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"\nKhởi tạo SFTTrainer với model: {args.model_name_or_path}")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    
    print("\nBắt đầu quá trình huấn luyện SFT...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    import shutil, glob
    for ckpt in glob.glob(f"{args.output_dir}/checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)
    print(f"Huấn luyện hoàn tất! Model lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()
