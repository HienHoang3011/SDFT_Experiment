"""Summarize a SAMSum comparison run from its saved artifacts."""

import argparse
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def fmt_score(value):
    return f"{100 * value:.2f}" if value is not None else "pending"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact_root", type=Path)
    args = parser.parse_args()
    root = args.artifact_root
    reports = root / "reports"
    manifest = read_json(reports / "fairness_manifest.json")
    if manifest is None:
        raise FileNotFoundError("fairness_manifest.json is missing")
    models = manifest["models"]
    benchmark = read_json(root / "stable-five" / "summary.json")
    kl = read_json(reports / "kl" / "summary.json")
    lines = [
        "# SAMSum method comparison",
        "",
        f"Run: `{manifest['run_name']}`. Base: Qwen2-7B. Dataset: SAMSum EMNLP 2019.",
        "",
        "## Configuration",
        "",
        f"All trainable methods use {manifest['shared_training']['epochs']} epochs, "
        f"learning rate {manifest['shared_training']['learning_rate']}, "
        f"effective batch size {manifest['shared_training']['per_device_batch_size'] * manifest['shared_training']['gradient_accumulation_steps']}, "
        f"and {manifest['shared_training']['optimizer']}. "
        f"Steer uses x_factor={manifest['steer']['x_factor']}, "
        f"giving a logit margin of log(1 + {manifest['steer']['x_factor']}). "
        "FAPM post-processes the Full SFT checkpoint.",
        "",
        "All trained methods use the same SAMSum train/dev splits, base checkpoint, seed, schedule, and optimizer. LoRA trains fewer parameters than full fine-tuning by design. FAPM is derived from Full SFT without another training pass; it shares Full SFT's data exposure. Scores compare final outcomes at the same epoch count, while GPU time and trainable parameter count differ by method.",
        "",
        "## SAMSum test ROUGE-L",
        "",
        "| Model | ROUGE-L (%) | Test examples |",
        "|---|---:|---:|",
    ]
    for model in models:
        payload = read_json(reports / model / "downstream_results.json")
        lines.append(f"| {model} | {fmt_score(payload['rougeL']) if payload else 'pending'} | {payload['num_samples'] if payload else 'pending'} |")
    lines += ["", "## Output distribution versus Base", "", "Mean KL over the full vocabulary, in nats per assistant token, on a fixed SAMSum dev probe.", "", "| Model | KL(base ∥ model) | KL(model ∥ base) | Tokens |", "|---|---:|---:|---:|"]
    for model in models:
        row = next((r for r in kl["rows"] if r["model"] == model), None) if kl else None
        lines.append(f"| {model} | {row['kl_base_to_model_nats_per_token']:.6f} | {row['kl_model_to_base_nats_per_token']:.6f} | {row['assistant_tokens']} |" if row else f"| {model} | pending | pending | pending |")
    lines += ["", "## Five capability benchmarks", "", "Full splits; scores are percentages. Per-example responses and scorer details are in `stable-five/models/`.", "", "| Benchmark | Metric | " + " | ".join(models) + " |", "|---|---|" + "---:|" * len(models)]
    task_specs = manifest["capability_tasks"]
    scores = {(r["task"], r["model"]): r for r in benchmark["rows"]} if benchmark else {}
    specs = benchmark["tasks"] if benchmark else {}
    for task in task_specs:
        metric = specs.get(task, {}).get("metric", "pending")
        cells = [fmt_score(scores[(task, model)]["score"]) if (task, model) in scores else "pending" for model in models]
        lines.append(f"| {task} | {metric} | " + " | ".join(cells) + " |")
    lines += ["", "## Artifact details", "", "The fairness manifest is in `reports/fairness_manifest.json`. SAMSum predictions and metrics are in `reports/<model>/`. KL measurements are in `reports/kl/`. Benchmark scores, scorer output, and sample responses are in `stable-five/`.", ""]
    destination = root / "REPORT.md"
    destination.write_text("\n".join(lines), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
