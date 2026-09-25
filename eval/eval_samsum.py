"""Evaluate SAMSum summaries with deterministic ROUGE-L."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", default="data/samsum_emnlp2019")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--max_input_length", type=int, default=2048)
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_path = config.get("base_model_name_or_path")
        if not base_path:
            raise ValueError("adapter_config.json is missing base_model_name_or_path")
        tokenizer_source = model_path if (Path(model_path) / "tokenizer_config.json").exists() else base_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, padding_side="left")
        configure_qwen2_base_tokenizer(tokenizer, tokenizer_source)
        base = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        configure_qwen2_base_tokenizer(tokenizer, model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sync_model_special_tokens(model, tokenizer)
    model.eval()
    return model, tokenizer


def extract_summary(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    text = re.sub(r"^\s*(?:summary|answer)\s*:\s*", "", text, flags=re.I)
    return text.strip()


def main() -> None:
    import torch
    from datasets import Dataset
    from rouge_score import rouge_scorer

    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise ValueError("batch_size and max_new_tokens must be positive")
    dataset = Dataset.load_from_disk(str(Path(args.data_root) / "test"))
    if len(dataset) != 819:
        raise AssertionError(f"Expected official SAMSum test split of 819, found {len(dataset)}")
    model, tokenizer = load_model_and_tokenizer(args.model_path)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rows, total = [], 0.0
    for start in range(0, len(dataset), args.batch_size):
        batch = dataset[start : start + args.batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in batch["prompt"]
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_input_length,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        decoded = tokenizer.batch_decode(
            generated[:, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        for offset, raw in enumerate(decoded):
            prediction = extract_summary(raw)
            reference = batch["summary"][offset]
            rouge_l = scorer.score(reference, prediction)["rougeL"].fmeasure
            total += rouge_l
            rows.append({
                "id": batch["id"][offset],
                "prediction": prediction,
                "reference": reference,
                "raw_response": raw,
                "rougeL": rouge_l,
            })
        print(f"[SAMSum ROUGE-L] {min(start + args.batch_size, len(dataset))}/{len(dataset)}", flush=True)
    result = {
        "task": "samsum_emnlp2019",
        "dataset": "Samsung/samsum",
        "split": "test",
        "num_samples": len(dataset),
        "primary_metric": "rougeL",
        "evaluation_method": "rouge_score.RougeScorer(rougeL, use_stemmer=True) f-measure",
        "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
        "rougeL": total / len(dataset),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "downstream_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    with (output_dir / "downstream_responses.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
