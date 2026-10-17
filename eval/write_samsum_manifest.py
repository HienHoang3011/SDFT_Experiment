"""Write the immutable configuration manifest for the SAMSum experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--learning_rate", required=True, type=float)
    parser.add_argument("--per_device_batch_size", required=True, type=int)
    parser.add_argument("--gradient_accumulation_steps", required=True, type=int)
    parser.add_argument("--max_sequence_length", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--steer_x_factor", required=True, type=float)
    parser.add_argument("--fapm_k_ratio", required=True, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "run_name": args.run_name,
        "dataset": "SAMSum (EMNLP 2019), mirror fastlayne/samsum",
        "models": ["base", "full_sft", "lora", "steer_lora", "steer_full", "fapm"],
        "shared_training": {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "per_device_batch_size": args.per_device_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_sequence_length": args.max_sequence_length,
            "optimizer": "AdamW 8-bit (bitsandbytes)",
            "scheduler": "cosine",
            "warmup_ratio": 0.03,
            "weight_decay": 0.1,
            "seed": args.seed,
        },
        "intended_difference": "Shared SAMSum train/dev schedule; FAPM post-processes Full SFT.",
        "steer": {"x_factor": args.steer_x_factor, "margin": "log(1 + x_factor)", "lora_and_full": True},
        "fapm": {"k_ratio": args.fapm_k_ratio, "source": "full_sft"},
        "lora_parameterization": {
            "rank": 256,
            "alpha": 512,
            "dropout": 0.3,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "capability_tasks": ["gsm8k_cot_zeroshot", "humaneval_instruct", "ifeval", "boolq", "bbh_cot_zeroshot"],
        "capability_runner": "benchmarks/stable_five/run.py, lm-eval 0.4.12, full splits",
        "downstream_metric": "SAMSum test ROUGE-L, greedy generation, max_new_tokens=128",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
