import argparse
import os
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens

def parse_args():
    parser = argparse.ArgumentParser(description="Elastic Weight Consolidation (EWC) Full Fine-Tuning")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--train_data_path", type=str, default="data/medqa_data/train_data")
    parser.add_argument("--dev_data_path", type=str, default="data/medqa_data/dev_data")
    parser.add_argument("--fisher_data_path", type=str, default=None, 
                        help="Dataset để tính Fisher. Nếu None, dùng Ma trận Đơn vị (L2 Regularization)")
    parser.add_argument("--output_dir", type=str, default="outputs/ewc-qwen2-7b")
    parser.add_argument("--ewc_lambda", type=float, default=100.0, help="Hệ số phạt EWC")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    return parser.parse_args()

def prepare_dataset(data_path):
    return load_from_disk(data_path)

class EWCTrainer(SFTTrainer):
    def __init__(self, ewc_lambda=100.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        
        # Lưu trọng số gốc trên CPU để tránh OOM (vì Full FT cho 7B rất tốn VRAM)
        print("[EWC] Đang sao chép trọng số gốc để làm mốc (anchor)...")
        self.original_weights = {
            n: p.clone().detach().to("cpu") 
            for n, p in self.model.named_parameters() if p.requires_grad
        }
        
        # Fisher Matrix. Hiện tại khởi tạo bằng 1 (Identity Matrix) -> tương đương L2 Regularization.
        # Để tính đúng Fisher, cần chạy 1 epoch backward trên prior data.
        print("[EWC] Đang khởi tạo Fisher Information Matrix (Mặc định = 1.0)...")
        self.fisher_matrix = {
            n: torch.ones_like(p, device="cpu") 
            for n, p in self.model.named_parameters() if p.requires_grad
        }

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # 1. Tính loss chuẩn (Cross Entropy) cho task mới
        loss_outputs = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        base_loss = loss_outputs[0] if return_outputs else loss_outputs
        
        # 2. Tính penalty EWC
        ewc_loss = 0.0
        for n, p in model.named_parameters():
            if p.requires_grad and n in self.original_weights:
                # Kéo dữ liệu từ CPU lên GPU hiện tại của tham số
                orig_w = self.original_weights[n].to(p.device, non_blocking=True)
                fisher = self.fisher_matrix[n].to(p.device, non_blocking=True)
                
                # Penalty = F * (theta - theta_star)^2
                ewc_loss += (fisher * (p - orig_w) ** 2).sum()
        
        # 3. Tổng hợp Loss
        total_loss = base_loss + (self.ewc_lambda / 2.0) * ewc_loss
        
        return (total_loss, loss_outputs[1]) if return_outputs else total_loss

def main():
    args = parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    configured_qwen2_base = configure_qwen2_base_tokenizer(
        tokenizer, args.model_name_or_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    print(f"[INFO] Load model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    if configured_qwen2_base:
        sync_model_special_tokens(model, tokenizer)
    model.config.use_cache = False
    
    train_dataset = prepare_dataset(args.train_data_path)
    eval_dataset = prepare_dataset(args.dev_data_path)
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        save_total_limit=1,
        eval_strategy="epoch",
        bf16=True,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
        average_tokens_across_devices=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        optim="adamw_torch",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
    )
    
    trainer = EWCTrainer(
        ewc_lambda=args.ewc_lambda,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )
    
    print("\n[INFO] BẮT ĐẦU HUẤN LUYỆN EWC FULL FINE-TUNING...")
    trainer.train()
    
    print(f"\n[INFO] Lưu model EWC tại: {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    import shutil, glob
    for ckpt in glob.glob(f"{args.output_dir}/checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)

if __name__ == "__main__":
    main()
