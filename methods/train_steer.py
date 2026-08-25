import argparse
import math
import torch
import torch.nn.functional as F
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

def build_steered_loss(x_factor):
    """Build a Trainer-compatible steered loss function."""
    if x_factor <= -1.0:
        raise ValueError("x_factor must be greater than -1.0")

    margin = math.log1p(x_factor)

    def steered_loss(outputs, labels, num_items_in_batch=None):
        if labels is None:
            raise ValueError("Steered loss requires labels")

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        flat_labels = shift_labels.reshape(-1)
        active_mask = flat_labels.ne(-100)

        # A fully masked micro-batch can occur after truncation. Return a zero
        # that stays connected to the graph so backward/DDP remain valid.
        if not active_mask.any():
            return logits.sum() * 0.0

        # Compute the KL in fp32: this loss can be very small once the margin is
        # satisfied, so bf16 precision is not sufficient here.
        active_logits = flat_logits[active_mask].float()
        active_labels = flat_labels[active_mask]

        with torch.no_grad():
            target_logits = active_logits.detach().clone()
            other_logits = target_logits.clone()
            row_indices = torch.arange(target_logits.size(0), device=target_logits.device)
            other_logits[row_indices, active_labels] = float("-inf")

            max_other_logits = other_logits.max(dim=-1).values
            current_gt_logits = target_logits[row_indices, active_labels]
            target_logits[row_indices, active_labels] = torch.maximum(
                current_gt_logits,
                max_other_logits + margin,
            )
            target_probs = F.softmax(target_logits, dim=-1)

        log_probs = F.log_softmax(active_logits, dim=-1)
        loss_sum = F.kl_div(log_probs, target_probs, reduction="sum")

        # During training, Transformers supplies the number of valid tokens in
        # the whole accumulated batch. Dividing by it makes gradient
        # accumulation equivalent to one large batch. Evaluation calls may not
        # supply it, in which case a local token mean is the correct reduction.
        if num_items_in_batch is None:
            denominator = active_mask.sum().to(device=loss_sum.device, dtype=loss_sum.dtype)
        else:
            denominator = torch.as_tensor(
                num_items_in_batch,
                device=loss_sum.device,
                dtype=loss_sum.dtype,
            )

        return loss_sum / denominator.clamp_min(1.0)

    return steered_loss

def parse_args():
    parser = argparse.ArgumentParser(description="Run Steered SFT Training")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train_data_path", type=str, default="data/medqa_data/train_data")
    parser.add_argument("--eval_data_path", type=str, default="data/medqa_data/eval_data")
    parser.add_argument("--dev_data_path", type=str, default="data/medqa_data/dev_data")
    parser.add_argument("--output_dir", type=str, default="outputs/steer-qwen2.5-7b")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--x_factor", type=float, default=1.0)
    parser.add_argument("--disable_peft", action="store_true", help="Vô hiệu hóa LoRA (Full Fine-Tuning)")
    return parser.parse_args()

def prepare_train_dataset(dataset_path):
    print(f"Loading train dataset from {dataset_path}")
    dataset = Dataset.load_from_disk(dataset_path)
    
    def format_messages(example):
        msgs = example.get("messages")
        output_text = str(example["output_text"])

        if isinstance(msgs, list) and msgs:
            msgs_copy = [dict(message) for message in msgs]
            if msgs_copy[-1].get("role") == "assistant":
                if str(msgs_copy[-1].get("content", "")) != output_text:
                    raise ValueError("The final assistant message does not match output_text")
            else:
                msgs_copy.append({"role": "assistant", "content": output_text})
            example["messages"] = msgs_copy
        else:
            example["messages"] = [
                {"role": "user", "content": str(example.get("prompt", ""))},
                {"role": "assistant", "content": output_text},
            ]
        return example
        
    dataset = dataset.map(format_messages, desc="Formatting conversation messages")
    
    cols_to_remove = [col for col in ["prompt", "text", "output_text"] if col in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
        
    return dataset

def main():
    args = parse_args()
    
    train_dataset = prepare_train_dataset(args.train_data_path)
    eval_dataset = prepare_train_dataset(args.dev_data_path)
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
        # A custom logits-based loss needs the regular LM head output. Keeping
        # TRL's default chunked_nll would patch a labels-only path that this
        # compute_loss_func cannot use.
        loss_type="nll",
        model_init_kwargs={"dtype": "bfloat16"},
        average_tokens_across_devices=True,
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
    
    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        compute_loss_func=build_steered_loss(args.x_factor),
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
