"""Full-weight SAMSum SFT with an L1 penalty on trainable parameters."""
from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from qwen2_base import (
    QWEN2_BASE_MODEL_ID,
    configure_qwen2_base_tokenizer,
    sync_model_special_tokens,
)
from train_sft import prepare_train_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=QWEN2_BASE_MODEL_ID)
    parser.add_argument("--train_data_path", default="data/samsum_emnlp2019/train")
    parser.add_argument("--dev_data_path", default="data/samsum_emnlp2019/dev")
    parser.add_argument("--output_dir", default="artifacts/manual/l1")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--l1_lambda",
        type=float,
        default=1e-8,
        help="Coefficient for lambda * sum(abs(trainable parameters)); tune on dev data.",
    )
    parser.add_argument("--model_dtype", choices=["float32", "bfloat16"], default="float32",
                        help="Stored weight dtype; pure bfloat16 rounds away small optimizer steps")
    args = parser.parse_args()
    if args.l1_lambda < 0:
        parser.error("l1_lambda must be nonnegative")
    return args


class L1Trainer(SFTTrainer):
    """SFTTrainer with sum-of-absolute-weights regularization during training."""

    def __init__(self, *, l1_lambda: float, **kwargs):
        super().__init__(**kwargs)
        self.l1_lambda = l1_lambda

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )
        base_loss = result[0] if return_outputs else result
        if not model.training or self.l1_lambda == 0:
            return result

        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not trainable:
            raise ValueError("L1 regularization requires trainable model parameters")
        penalty_device = trainable[0].device
        penalty = sum(
            (parameter.float().abs().sum().to(penalty_device) for parameter in trainable),
            torch.zeros((), dtype=torch.float32, device=penalty_device),
        )

        # When Trainer uses a global-token denominator, it does not apply the
        # usual gradient-accumulation division to the loss, so scale the
        # parameter-only penalty to contribute once per optimizer update.
        accumulation = (
            self.current_gradient_accumulation_steps
            if num_items_in_batch is not None and self.model_accepts_loss_kwargs
            else 1
        )
        loss = base_loss + self.l1_lambda * penalty / accumulation
        return (loss, result[1]) if return_outputs else loss


def main():
    args = parse_args()
    train_dataset = prepare_train_dataset(args.train_data_path)
    dev_dataset = prepare_train_dataset(args.dev_data_path)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        torch_dtype=getattr(torch, args.model_dtype),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    configure_qwen2_base_tokenizer(tokenizer, args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sync_model_special_tokens(model, tokenizer)
    model.config.use_cache = False

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
        optim="adamw_torch",
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
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = L1Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        l1_lambda=args.l1_lambda,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
