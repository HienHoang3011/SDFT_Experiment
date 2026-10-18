import argparse
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from qwen2_base import (
    QWEN2_BASE_MODEL_ID,
    configure_qwen2_base_tokenizer,
    sync_model_special_tokens,
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run full SFT training on SAMSum")
    parser.add_argument("--model_name_or_path", type=str, default=QWEN2_BASE_MODEL_ID,
                        help="Path to the model or HF model name")
    parser.add_argument("--train_data_path", type=str, default="data/samsum_emnlp2019/train",
                        help="Path to the train dataset (HF disk format)")
    parser.add_argument("--dev_data_path", type=str, default="data/samsum_emnlp2019/dev",
                        help="Path to the dev dataset for validation during training")
    parser.add_argument("--output_dir", type=str, default="artifacts/manual/full_sft",
                        help="Directory to save the trained model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per GPU for training")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1,
                        help="Batch size per GPU for evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64,
                        help="Number of update steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=2,
                        help="Total number of training epochs")
    parser.add_argument("--max_seq_length", type=int, default=2048, # max_prompt_length (1024) + max_completion_length (1024)
                        help="Maximum sequence length")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic data ordering")
    parser.add_argument("--model_dtype", choices=["float32", "bfloat16"], default="float32",
                        help="Stored weight dtype. float32 keeps master weights; bf16=True still "
                             "autocasts compute. Pure bfloat16 rounds away lr=1e-5 updates on |w| >= 2^-8.")
    parser.add_argument("--optim", type=str, default="adamw_8bit",
                        help="Optimizer; paged_adamw_8bit is the same update with paged states")
    parser.add_argument("--max_steps", type=int, default=-1, help="Optional step limit for smoke checks")
    parser.add_argument("--freeze_layers", type=str, default=None,
                        help="Transformer layers to freeze, e.g. '0-6' or '21-27' (0-based, inclusive)")
    parser.add_argument("--freeze_embeddings", action="store_true", help="Also freeze embed_tokens")
    parser.add_argument("--freeze_head", action="store_true", help="Also freeze the final norm and lm_head")
    parser.add_argument("--save_dtype", choices=["float32", "bfloat16"], default="float32",
                        help="dtype of the saved checkpoint; bfloat16 halves disk and matches bf16 evaluation")
    return parser.parse_args()


def freeze_parameters(model, layers, embeddings, head):
    """Freeze the requested blocks; returns (frozen, total) parameter counts."""
    frozen_layers = set()
    if layers:
        start, end = (int(x) for x in layers.split("-"))
        frozen_layers = set(range(start, end + 1))
    for name, param in model.named_parameters():
        parts = name.split(".")
        in_layer = "layers" in parts and int(parts[parts.index("layers") + 1]) in frozen_layers
        is_embed = embeddings and "embed_tokens" in name
        is_head = head and (name.startswith("lm_head") or name == "model.norm.weight")
        if in_layer or is_embed or is_head:
            param.requires_grad_(False)
    if embeddings:
        # With gradient checkpointing, a frozen embedding output carries no grad and the
        # checkpointed blocks above it would receive none either.
        model.enable_input_require_grads()
    total = sum(p.numel() for p in model.parameters())
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return frozen, total

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
    
    print(f"\n[INFO] Số lượng tập train: {len(train_dataset)}")
    print(f"[INFO] Số lượng tập dev (validation): {len(eval_dataset)}")
    
    print("\n[Kiểm tra Dữ liệu Train mẫu]")
    print(str(train_dataset[0]["messages"])[:500] + "...")
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
        average_tokens_across_devices=True,
        
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        optim=args.optim,
        max_steps=args.max_steps,
        max_grad_norm=1.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="no",
        report_to="none",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
    )
    
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=getattr(torch, args.model_dtype),
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    configured_qwen2_base = configure_qwen2_base_tokenizer(
        tokenizer, args.model_name_or_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if configured_qwen2_base:
        sync_model_special_tokens(model, tokenizer)
    model.config.use_cache = False
    if args.freeze_layers or args.freeze_embeddings or args.freeze_head:
        frozen, total = freeze_parameters(model, args.freeze_layers, args.freeze_embeddings, args.freeze_head)
        print(f"[INFO] Frozen {frozen:,} / {total:,} parameters ({100 * frozen / total:.1f}%): "
              f"layers={args.freeze_layers} embeddings={args.freeze_embeddings} head={args.freeze_head}")
    print(f"\nKhởi tạo SFTTrainer với model: {args.model_name_or_path}")
    
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    print("\nBắt đầu quá trình huấn luyện SFT...")
    trainer.train()
    
    if args.save_dtype == "bfloat16":
        trainer.model.to(torch.bfloat16)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    import shutil, glob
    for ckpt in glob.glob(f"{args.output_dir}/checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)
    print(f"Huấn luyện hoàn tất! Model lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()
