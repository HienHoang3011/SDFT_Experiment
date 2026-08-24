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
            forward_inputs = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**forward_inputs)
            
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
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
    parser.add_argument("--train_data_path", type=str, default="data/medqa_data/train_data")
    parser.add_argument("--eval_data_path", type=str, default="data/medqa_data/eval_data")
    parser.add_argument("--dev_data_path", type=str, default="data/medqa_data/dev_data")
    parser.add_argument("--output_dir", type=str, default="outputs/steer-qwen2.5-7b")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--disable_peft", action="store_true", help="Vô hiệu hóa LoRA (Full Fine-Tuning)")
    return parser.parse_args()

def prepare_train_dataset(dataset_path, model_name_or_path):
    print(f"Loading train dataset from {dataset_path}")
    dataset = Dataset.load_from_disk(dataset_path)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    def format_messages(example):
        msgs = example.get("messages", example.get("prompt", []))
        if isinstance(msgs, list):
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
        gradient_checkpointing=True,
        optim="paged_adamw_8bit"
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
    
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    trainer_class = _build_steered_trainer_class(SFTTrainer)
    trainer = trainer_class(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        x_factor=1.0
    )
    
    print("\nBắt đầu quá trình huấn luyện Steered SFT...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    import shutil, glob
    for ckpt in glob.glob(f"{args.output_dir}/checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)
    print(f"Huấn luyện hoàn tất! Model lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()
