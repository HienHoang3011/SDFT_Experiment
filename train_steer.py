import argparse
import os
import math
import torch
import torch.nn.functional as F
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def _build_steered_trainer_class(sft_trainer_cls):
    class SteeredTrainer(sft_trainer_cls):
        def __init__(self, *args, x_factor=0.2, **kwargs):
            super().__init__(*args, **kwargs)
            self.margin = math.log(1 + x_factor)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(**inputs)
            logits = outputs.get("logits")
            labels = inputs.get("labels")

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels = shift_labels.view(-1)

            mask = shift_labels != -100
            active_logits = shift_logits[mask]
            active_labels = shift_labels[mask]

            with torch.no_grad():
                target_logits = active_logits.clone()
                temp_logits = target_logits.clone()
                batch_idx = torch.arange(target_logits.size(0), device=target_logits.device)
                temp_logits[batch_idx, active_labels] = float('-inf')

                max_other_logits, _ = torch.max(temp_logits, dim=-1)
                current_gt_logits = target_logits[batch_idx, active_labels]

                target_gt_logits = torch.max(current_gt_logits, max_other_logits + self.margin)
                target_logits[batch_idx, active_labels] = target_gt_logits

                target_probs = F.softmax(target_logits, dim=-1)

            log_probs = F.log_softmax(active_logits, dim=-1)
            loss = F.kl_div(log_probs, target_probs, reduction='batchmean')

            return (loss, outputs) if return_outputs else loss
    return SteeredTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Run Steered SFT Training")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_data_path", type=str, default="data/science_data/train_data")
    parser.add_argument("--eval_data_path", type=str, default="data/science_data/eval_data")
    parser.add_argument("--output_dir", type=str, default="outputs/steer-qwen2.5-7b")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=1536)
    parser.add_argument("--disable_peft", action="store_true", help="Vô hiệu hóa LoRA (Full Fine-Tuning)")
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
            msgs_copy = list(msgs)
            msgs_copy.append({"role": "assistant", "content": example["output_text"]})
            example["text"] = tokenizer.apply_chat_template(
                msgs_copy, tokenize=False, add_generation_prompt=False
            )
        else:
            example["text"] = str(msgs) + "\n" + str(example["output_text"])
        return example
        
    dataset = dataset.map(format_text, desc="Formatting conversation text")
    return dataset

def main():
    args = parse_args()
    
    full_train_dataset = prepare_train_dataset(args.train_data_path, args.model_name_or_path)
    split_dataset = full_train_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
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
    
    trainer_class = _build_steered_trainer_class(SFTTrainer)
    trainer = trainer_class(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        x_factor=1.0  # According to user's snippet
    )
    
    print("\nBắt đầu quá trình huấn luyện Steered SFT...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    print(f"Huấn luyện hoàn tất! Model lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()
