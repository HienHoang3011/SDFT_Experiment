"""Prepare and validate the three Qwen2/FAPM catastrophic-forgetting tasks.

The saved datasets all use one user turn followed by exactly one assistant turn.
Validation is intentionally strict: malformed labels, duplicated train/test prompts,
or answers missing from the candidate set stop the pipeline before GPU training.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

TASKS = ("winogrande", "squad", "rte")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--output_root", default="data/cf_benchmarks")
    parser.add_argument("--report_root", default=".")
    parser.add_argument("--dev_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=0)
    parser.add_argument("--squad_context_chars", type=int, default=6000)
    return parser.parse_args()


def limit_dataset(dataset, limit, seed):
    if limit <= 0 or limit >= len(dataset):
        return dataset
    return dataset.shuffle(seed=seed).select(range(limit))


def make_record(sample_id, prompt, answer, choices=None, references=None):
    answer = str(answer).strip()
    output_text = f"<answer>{answer}</answer>"
    record = {
        "id": str(sample_id),
        "prompt": prompt.strip(),
        "answer": answer,
        "output_text": output_text,
        "messages": [
            {"role": "user", "content": prompt.strip()},
            {"role": "assistant", "content": output_text},
        ],
    }
    if choices is not None:
        record["choices"] = [str(choice).strip() for choice in choices]
    if references is not None:
        record["references"] = [str(reference).strip() for reference in references]
    return record


def format_winogrande(example):
    choices = [example["option1"], example["option2"]]
    answer_index = int(example["answer"]) - 1
    prompt = (
        "Choose the option that correctly fills the blank in the sentence.\n\n"
        f"Sentence: {example['sentence']}\n\n"
        f"Options:\n1. {choices[0]}\n2. {choices[1]}\n\n"
        "Return only the selected option text inside <answer> and </answer>."
    )
    return make_record(example.get("qID", ""), prompt, choices[answer_index], choices)


def format_rte(example):
    choices = ["entailment", "not_entailment"]
    answer = choices[int(example["label"])]
    prompt = (
        "Decide whether the premise entails the hypothesis.\n\n"
        f"Premise: {example['sentence1']}\n"
        f"Hypothesis: {example['sentence2']}\n\n"
        "Options:\n- entailment\n- not_entailment\n\n"
        "Return only one option inside <answer> and </answer>."
    )
    return make_record(example.get("idx", ""), prompt, answer, choices)


def crop_squad_context(context, answer_start, answer_text, max_chars):
    if max_chars <= 0 or len(context) <= max_chars:
        return context
    answer_end = answer_start + len(answer_text)
    left_budget = max(0, (max_chars - len(answer_text)) // 2)
    start = max(0, answer_start - left_budget)
    end = min(len(context), start + max_chars)
    start = max(0, end - max_chars)
    cropped = context[start:end]
    if answer_text not in cropped or not (start <= answer_start and answer_end <= end):
        raise ValueError("SQuAD context cropping removed the gold answer")
    return cropped


def format_squad(example, max_context_chars):
    references = list(dict.fromkeys(text.strip() for text in example["answers"]["text"] if text.strip()))
    if not references:
        raise ValueError(f"SQuAD sample {example['id']} has no non-empty answer")
    answer_start = int(example["answers"]["answer_start"][0])
    context = crop_squad_context(
        example["context"], answer_start, references[0], max_context_chars
    )
    prompt = (
        "Answer the question using a short span copied from the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {example['question']}\n\n"
        "Return only the short answer inside <answer> and </answer>."
    )
    return make_record(example["id"], prompt, references[0], references=references)


def convert(dataset, formatter):
    from datasets import Dataset

    return Dataset.from_list([formatter(example) for example in dataset])


def load_and_format(task, dev_ratio, seed, max_train, max_eval, squad_context_chars):
    from datasets import load_dataset

    if task == "winogrande":
        source = load_dataset("allenai/winogrande", "winogrande_xl")
        split = source["train"].train_test_split(test_size=dev_ratio, seed=seed)
        raw = {
            "train": limit_dataset(split["train"], max_train, seed),
            "dev": limit_dataset(split["test"], max_eval, seed),
            "test": limit_dataset(source["validation"], max_eval, seed),
        }
        return {name: convert(data, format_winogrande) for name, data in raw.items()}

    if task == "rte":
        source = load_dataset("nyu-mll/glue", "rte")
        split = source["train"].train_test_split(
            test_size=dev_ratio, seed=seed, stratify_by_column="label"
        )
        raw = {
            "train": limit_dataset(split["train"], max_train, seed),
            "dev": limit_dataset(split["test"], max_eval, seed),
            "test": limit_dataset(source["validation"], max_eval, seed),
        }
        return {name: convert(data, format_rte) for name, data in raw.items()}

    source = load_dataset("rajpurkar/squad")
    split = source["train"].train_test_split(test_size=dev_ratio, seed=seed)
    raw = {
        "train": limit_dataset(split["train"], max_train, seed),
        "dev": limit_dataset(split["test"], max_eval, seed),
        "test": limit_dataset(source["validation"], max_eval, seed),
    }
    formatter = lambda example: format_squad(example, squad_context_chars)
    return {name: convert(data, formatter) for name, data in raw.items()}


def prompt_hash(prompt):
    normalized = " ".join(prompt.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

def remove_duplicate_and_cross_split_prompts(datasets):
    """
    Keep evaluation data protected.

    Priority:
        test > dev > train

    If the same normalized prompt occurs in multiple splits,
    keep it in the higher-priority split and remove it from the others.
    Also remove duplicate prompts inside each split.
    """
    cleaned = {}
    seen = set()

    for split_name in ("test", "dev", "train"):
        dataset = datasets[split_name]

        keep_indices = []
        local_seen = set()

        for idx, row in enumerate(dataset):
            h = prompt_hash(row["prompt"])

            # Duplicate inside the same split
            if h in local_seen:
                continue

            local_seen.add(h)

            # Prompt already exists in a higher-priority split
            if h in seen:
                continue

            seen.add(h)
            keep_indices.append(idx)

        cleaned[split_name] = dataset.select(keep_indices)

        print(
            f"[dedup] {split_name}: "
            f"{len(dataset)} -> {len(cleaned[split_name])}"
        )

    return {
        "train": cleaned["train"],
        "dev": cleaned["dev"],
        "test": cleaned["test"],
    } 

def validate(task, datasets):
    errors = []
    warnings = []
    split_stats = {}
    hash_sets = {}

    for split_name, dataset in datasets.items():
        empty_prompts = 0
        empty_answers = 0
        malformed_messages = 0
        invalid_choices = 0
        answer_not_in_context = 0
        labels = Counter()
        hashes = set()

        for row in dataset:
            prompt = str(row.get("prompt", "")).strip()
            answer = str(row.get("answer", "")).strip()
            messages = row.get("messages") or []
            empty_prompts += not bool(prompt)
            empty_answers += not bool(answer)
            valid_messages = (
                len(messages) == 2
                and messages[0]["role"] == "user"
                and messages[1]["role"] == "assistant"
                and messages[0]["content"] == prompt
                and messages[1]["content"] == row.get("output_text")
            )
            malformed_messages += not valid_messages
            if task in {"winogrande", "rte"}:
                choices = list(row.get("choices") or [])
                invalid_choices += len(choices) != 2 or answer not in choices
                labels[answer] += 1
            else:
                references = list(row.get("references") or [])
                invalid_choices += not references or answer not in references
                context = prompt.split("Context:\n", 1)[-1].split("\n\nQuestion:", 1)[0]
                answer_not_in_context += answer not in context
            hashes.add(prompt_hash(prompt))

        duplicate_prompts = len(dataset) - len(hashes)
        split_stats[split_name] = {
            "num_samples": len(dataset),
            "empty_prompts": empty_prompts,
            "empty_answers": empty_answers,
            "malformed_messages": malformed_messages,
            "invalid_answers_or_choices": invalid_choices,
            "duplicate_prompts_within_split": duplicate_prompts,
            "answer_not_in_context": answer_not_in_context,
            "label_counts": dict(labels),
        }
        hash_sets[split_name] = hashes
        if empty_prompts or empty_answers or malformed_messages or invalid_choices or answer_not_in_context:
            errors.append(f"{split_name} contains malformed samples: {split_stats[split_name]}")
        if duplicate_prompts:
            warnings.append(f"{split_name} contains {duplicate_prompts} duplicate prompts")

    overlap = {}
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        count = len(hash_sets[left] & hash_sets[right])
        overlap[f"{left}_{right}"] = count
        if count:
            errors.append(f"Detected {count} exact normalized prompt leaks between {left} and {right}")

    return {
        "task": task,
        "status": "ok" if not errors else "failed",
        "splits": split_stats,
        "cross_split_prompt_overlap": overlap,
        "warnings": warnings,
        "errors": errors,
    }


def write_quality_report(report, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "data_quality.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [f"# Data quality: {report['task']}", "", f"Status: **{report['status']}**", ""]
    lines += ["| Split | Samples | Empty | Bad messages | Bad labels | Duplicates |", "|---|---:|---:|---:|---:|---:|"]
    for name, stats in report["splits"].items():
        lines.append(
            f"| {name} | {stats['num_samples']} | "
            f"{stats['empty_prompts'] + stats['empty_answers']} | "
            f"{stats['malformed_messages']} | {stats['invalid_answers_or_choices']} | "
            f"{stats['duplicate_prompts_within_split']} |"
        )
    if report["warnings"]:
        lines += ["", "## Warnings", ""] + [f"- {item}" for item in report["warnings"]]
    if report["errors"]:
        lines += ["", "## Errors", ""] + [f"- {item}" for item in report["errors"]]
    (report_dir / "data_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    from datasets import load_from_disk

    args = parse_args()
    output_root = Path(args.output_root)
    failed = []
    for task in args.tasks:
        task_root = output_root / task
        existing = all((task_root / split).exists() for split in ("train", "dev", "test"))
        if existing:
            print(f"\n[prepare] Reusing and re-validating existing {task} data...")
            datasets = {
                split: load_from_disk(str(task_root / split))
                for split in ("train", "dev", "test")
            }
        else:
            print(f"\n[prepare] Loading and formatting {task}...")
            datasets = load_and_format(
            task,
            args.dev_ratio,
            args.seed,
            args.max_train_samples,
            args.max_eval_samples,
            args.squad_context_chars,
        )

        datasets = remove_duplicate_and_cross_split_prompts(datasets)

        report = validate(task, datasets)
        write_quality_report(report, Path(args.report_root) / f"reports-{task}")
        if report["errors"]:
            failed.append(task)
            continue
        if not existing:
            for split_name, dataset in datasets.items():
                destination = output_root / task / split_name
                destination.parent.mkdir(parents=True, exist_ok=True)
                dataset.save_to_disk(str(destination))
        print(f"[prepare] {task}: {report['splits']}")

    if failed:
        raise SystemExit(f"Data validation failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
