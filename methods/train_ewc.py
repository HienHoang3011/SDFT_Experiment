"""Full-weight SAMSum SFT with diagonal empirical-Fisher EWC."""
from __future__ import annotations

import argparse
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from qwen2_base import QWEN2_BASE_MODEL_ID, configure_qwen2_base_tokenizer, sync_model_special_tokens
from train_sft import prepare_train_dataset


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_name_or_path", default=QWEN2_BASE_MODEL_ID)
    parser.add_argument("--train_data_path", default="data/samsum_emnlp2019/train")
    parser.add_argument("--dev_data_path", default="data/samsum_emnlp2019/dev")
    parser.add_argument("--fisher_max_samples", type=int, default=256)
    parser.add_argument("--ewc_lambda", type=float, default=100.0)
    parser.add_argument("--output_dir", default="artifacts/manual/ewc")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.fisher_max_samples <= 0 or args.ewc_lambda < 0:
        parser.error("fisher_max_samples must be positive and ewc_lambda nonnegative")
    return args


def fisher_inputs(example, tokenizer, max_length):
    """Build SAMSum training inputs and mask every token outside assistant output."""
    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
        )
        ids = list(encoded["input_ids"])[:max_length]
        assistant_mask = list(encoded.get("assistant_masks", []))[:max_length]
        if len(assistant_mask) != len(ids) or not any(assistant_mask):
            raise ValueError("Chat template did not return a valid assistant-token mask for Fisher")
        labels = [token_id if active else -100 for token_id, active in zip(ids, assistant_mask)]
        return ids, labels

    # Keep support for simple tokenized/text datasets used by small CPU checks.
    if "input_ids" in example:
        ids = list(example["input_ids"])[:max_length]
    elif isinstance(example.get("text"), str):
        ids = tokenizer(example["text"], truncation=True, max_length=max_length)["input_ids"]
    else:
        raise ValueError("Fisher examples need SAMSum messages, input_ids, or text")
    return ids, ids


def estimate_empirical_fisher(model, tokenizer, dataset, max_samples, max_length, seed):
    """Estimate diagonal Fisher at the initial weights from assistant-only train loss.

    Each sample contributes the square of its mean assistant-token NLL gradient;
    averaging these per-example squares is the usual empirical-Fisher estimate.
    """
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    fisher = {
        name: torch.zeros_like(parameter, dtype=torch.float32, device="cpu")
        for name, parameter in trainable
    }
    selected = random.Random(seed).sample(range(len(dataset)), min(max_samples, len(dataset)))
    model.eval()
    valid = 0
    input_device = next(model.parameters()).device
    for index in selected:
        ids, labels = fisher_inputs(dataset[index], tokenizer, max_length)
        if len(ids) < 2 or not any(label != -100 for label in labels[1:]):
            continue
        input_ids = torch.tensor([ids], dtype=torch.long, device=input_device)
        target_ids = torch.tensor([labels], dtype=torch.long, device=input_device)
        model.zero_grad(set_to_none=True)
        loss = model(input_ids=input_ids, labels=target_ids, use_cache=False).loss
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite Fisher loss at train sample {index}")
        loss.backward()
        for name, parameter in trainable:
            if parameter.grad is not None:
                fisher[name].add_(parameter.grad.detach().float().cpu().square())
        valid += 1
    model.zero_grad(set_to_none=True)
    model.train()
    if valid == 0:
        raise ValueError("No SAMSum train examples with assistant targets for Fisher")
    for value in fisher.values():
        value.div_(valid)
    return fisher, valid


class EWCQuadraticPenalty(torch.autograd.Function):
    """Compute EWC value/gradient without retaining a second 7B-parameter graph."""

    @staticmethod
    def forward(ctx, *args):
        *parameters, anchors, fishers = args
        ctx.save_for_backward(*parameters)
        ctx.anchors = anchors
        ctx.fishers = fishers
        total = torch.zeros((), dtype=torch.float32, device=parameters[0].device)
        for parameter, anchor, fisher in zip(parameters, anchors, fishers):
            delta = parameter.detach().float() - anchor.to(parameter.device)
            term = (fisher.to(parameter.device) * delta.square()).sum()
            total = total + term.to(total.device)
        return total

    @staticmethod
    def backward(ctx, grad_output):
        gradients = []
        for parameter, anchor, fisher in zip(ctx.saved_tensors, ctx.anchors, ctx.fishers):
            delta = parameter.detach().float() - anchor.to(parameter.device)
            gradient = 2 * fisher.to(parameter.device) * delta
            gradients.append((gradient * grad_output.to(parameter.device)).to(parameter.dtype))
        return (*gradients, None, None)


class EWCTrainer(SFTTrainer):
    def __init__(self, *, fisher, anchor, ewc_lambda, **kwargs):
        super().__init__(**kwargs)
        self.fisher = fisher
        self.anchor = anchor
        self.ewc_lambda = ewc_lambda
        names = {name for name, parameter in self.model.named_parameters() if parameter.requires_grad}
        if names != set(fisher) or names != set(anchor):
            raise ValueError("EWC Fisher/anchor parameters do not match the trainable model")

    def ewc_penalty(self, model):
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("EWC model has no trainable parameters")
        parameters = [parameter for _, parameter in trainable]
        anchors = tuple(self.anchor[name] for name, _ in trainable)
        fishers = tuple(self.fisher[name] for name, _ in trainable)
        return EWCQuadraticPenalty.apply(*parameters, anchors, fishers)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        result = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        base_loss = result[0] if return_outputs else result
        if not model.training:
            return result
        penalty = self.ewc_penalty(model)
        # With a global-token denominator, Trainer does not divide the loss by
        # gradient accumulation. Apply the regularizer once per optimizer step.
        accumulation = (
            self.current_gradient_accumulation_steps
            if num_items_in_batch is not None and self.model_accepts_loss_kwargs
            else 1
        )
        loss = base_loss + self.ewc_lambda * penalty / (2 * accumulation)
        return (loss, result[1]) if return_outputs else loss


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    configure_qwen2_base_tokenizer(tokenizer, args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16
    )
    sync_model_special_tokens(model, tokenizer)
    model.config.use_cache = False
    train_dataset = prepare_train_dataset(args.train_data_path)
    anchor = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    fisher, used = estimate_empirical_fisher(
        model, tokenizer, train_dataset, args.fisher_max_samples, args.max_seq_length, args.seed
    )
    print(f"[EWC] Empirical Fisher from {used} SAMSum train examples (assistant tokens only)", flush=True)
    dev_dataset = prepare_train_dataset(args.dev_data_path)
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
    trainer = EWCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        fisher=fisher,
        anchor=anchor,
        ewc_lambda=args.ewc_lambda,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
