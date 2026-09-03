"""Evaluate WinoGrande/RTE with conditional likelihood and SQuAD with EM/F1."""

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("winogrande", "squad", "rte"))
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_root", default="data/cf_benchmarks")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--max_input_length", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_model_and_tokenizer(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        config = json.loads(adapter_config.read_text(encoding="utf-8"))
        base_path = config.get("base_model_name_or_path")
        if not base_path:
            raise ValueError("adapter_config.json has no base_model_name_or_path")
        tokenizer_source = model_path if (Path(model_path) / "tokenizer_config.json").exists() else base_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, padding_side="left")
        configure_qwen2_base_tokenizer(tokenizer, tokenizer_source)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, device_map="auto"
        )
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, model_path)
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


def score_choices(model, tokenizer, prompt, choices):
    """Length-normalized conditional log-likelihood of each answer value.

    Only answer-value tokens are scored. Shared ChatML and <answer> tokens do
    not make a base model look weak merely because it has not learned the XML
    formatting yet.
    """
    import torch
    import torch.nn.functional as F

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    prefix_ids = tokenizer(
        rendered + "<answer>", add_special_tokens=False
    )["input_ids"]
    choice_ids = [
        tokenizer(str(choice), add_special_tokens=False)["input_ids"] for choice in choices
    ]
    if any(not ids for ids in choice_ids):
        raise ValueError("A candidate answer tokenized to an empty sequence")

    sequences = [prefix_ids + ids for ids in choice_ids]
    max_length = max(len(ids) for ids in sequences)
    pad_id = tokenizer.pad_token_id
    input_rows, attention_rows, label_rows = [], [], []
    for sequence, answer_ids in zip(sequences, choice_ids):
        pad = max_length - len(sequence)
        input_rows.append([pad_id] * pad + sequence)
        attention_rows.append([0] * pad + [1] * len(sequence))
        label_rows.append(
            [-100] * (pad + len(prefix_ids)) + answer_ids
        )

    device = model.device
    input_ids = torch.tensor(input_rows, device=device)
    attention_mask = torch.tensor(attention_rows, device=device)
    labels = torch.tensor(label_rows, device=device)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    token_log_probs = F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (
        (token_log_probs * mask).sum(-1) / mask.sum(-1).clamp_min(1)
    ).tolist()


def normalize_answer(text):
    def remove_articles(value):
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def remove_punctuation(value):
        return "".join(character for character in value if character not in string.punctuation)

    return " ".join(remove_articles(remove_punctuation(text.lower())).split())


def exact_match(prediction, reference):
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction, reference):
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = Counter(prediction_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def extract_short_answer(text):
    match = re.search(r"<answer>(.*?)</answer>", text, flags=re.I | re.S)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>(.*)", text, flags=re.I | re.S)
    if match:
        text = match.group(1)
    text = re.sub(r"^(the\s+)?answer\s+is\s*:?\s*", "", text.strip(), flags=re.I)
    return text.splitlines()[0].strip().strip("<> ") if text.strip() else ""


def evaluate_classification(model, tokenizer, dataset):
    rows = []
    correct = 0
    for index, sample in enumerate(dataset):
        choices = list(sample["choices"])
        scores = score_choices(model, tokenizer, sample["prompt"], choices)
        prediction = choices[max(range(len(scores)), key=scores.__getitem__)]
        is_correct = prediction == sample["answer"]
        correct += is_correct
        rows.append(
            {
                "id": sample["id"],
                "prediction": prediction,
                "answer": sample["answer"],
                "choices": choices,
                "choice_mean_logprobs": scores,
                "correct": bool(is_correct),
            }
        )
        if (index + 1) % 100 == 0:
            print(f"[eval] {index + 1}/{len(dataset)}")
    return {"accuracy": correct / max(1, len(dataset))}, rows


def evaluate_squad(model, tokenizer, dataset, batch_size, max_new_tokens, max_input_length):
    import torch

    rows = []
    exact_total = 0.0
    f1_total = 0.0
    for start in range(0, len(dataset), batch_size):
        batch = dataset[start : start + batch_size]
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
            max_length=max_input_length,
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for offset, raw_response in enumerate(decoded):
            references = list(batch["references"][offset])
            prediction = extract_short_answer(raw_response)
            em = max(exact_match(prediction, reference) for reference in references)
            f1 = max(token_f1(prediction, reference) for reference in references)
            exact_total += em
            f1_total += f1
            rows.append(
                {
                    "id": batch["id"][offset],
                    "prediction": prediction,
                    "raw_response": raw_response,
                    "references": references,
                    "exact_match": em,
                    "f1": f1,
                }
            )
        print(f"[eval] {min(start + batch_size, len(dataset))}/{len(dataset)}")
    denominator = max(1, len(dataset))
    return {"f1": f1_total / denominator, "exact_match": exact_total / denominator}, rows


def main():
    from datasets import Dataset

    args = parse_args()
    dataset = Dataset.load_from_disk(str(Path(args.data_root) / args.task / "test"))
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    model, tokenizer = load_model_and_tokenizer(args.model_path)

    if args.task in {"winogrande", "rte"}:
        metrics, rows = evaluate_classification(model, tokenizer, dataset)
        primary_metric = "accuracy"
        method = "length-normalized conditional answer likelihood"
    else:
        metrics, rows = evaluate_squad(
            model,
            tokenizer,
            dataset,
            args.batch_size,
            args.max_new_tokens,
            args.max_input_length,
        )
        primary_metric = "f1"
        method = "official-style normalized SQuAD token F1 and exact match"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "task": args.task,
        "model_path": args.model_path,
        "num_samples": len(dataset),
        "primary_metric": primary_metric,
        "evaluation_method": method,
        **metrics,
    }
    (output_dir / "downstream_results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output_dir / "downstream_responses.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
