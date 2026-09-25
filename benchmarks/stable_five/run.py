"""Reproduce the five retained SAMSum 2-epoch capability benchmarks."""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.metadata
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "artifacts/samsum/stable-2epoch/stable-five"
LM_EVAL_VERSION = "0.4.12"
MODELS = {
    "base": "Qwen/Qwen2-7B",
    "full_sft": str(ROOT / "artifacts/samsum/stable-2epoch/models/full_sft"),
    "lora": str(ROOT / "artifacts/samsum/stable-2epoch/models/lora"),
}
TASKS = {
    "gsm8k_cot_zeroshot": {
        "metric": "exact_match,flexible-extract",
        "num_fewshot": 0,
        "batch_size": 1,
        "gen_kwargs": None,
        "expected_samples": 1319,
    },
    "humaneval_instruct": {
        "metric": "pass@1,create_test",
        "num_fewshot": None,
        "batch_size": 4,
        "gen_kwargs": None,
        "expected_samples": 164,
    },
    "ifeval": {
        "metric": "prompt_level_strict_acc,none",
        "num_fewshot": None,
        "batch_size": 4,
        "gen_kwargs": None,
        "expected_samples": 541,
    },
    "boolq": {
        "metric": "acc,none",
        "num_fewshot": None,
        "batch_size": 4,
        "gen_kwargs": None,
        "expected_samples": 3270,
    },
    "bbh_cot_zeroshot": {
        "metric": "exact_match,flexible-extract",
        "num_fewshot": None,
        "batch_size": 4,
        "gen_kwargs": {"max_gen_toks": 2048, "do_sample": False},
        "expected_samples": 6511,
    },
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", action="append", default=[], metavar="NAME=PATH",
                        help="Add or override a model for a common five-task evaluation")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.model_paths = dict(MODELS)
    for specification in args.model_path:
        name, separator, path = specification.partition("=")
        if not separator or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or not path:
            parser.error("--model-path must be NAME=PATH with a safe lowercase name")
        args.model_paths[name] = path
    if args.models is None:
        args.models = list(args.model_paths)
    if any(name not in args.model_paths for name in args.models):
        parser.error("--models contains a name without a configured model path")
    if args.worker and args.worker not in args.model_paths:
        parser.error("--worker has no configured model path")
    return args


def json_default(value: Any) -> Any:
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def sample_count(result: dict[str, Any], task: str) -> int:
    counts = result["n-samples"]
    if task in counts:
        count = counts[task]
        if count["original"] != count["effective"]:
            raise ValueError(f"Incomplete split for {task}: {count}")
        return int(count["effective"])
    children = [value for name, value in counts.items() if name.startswith(task + "_")]
    if not children or any(value["original"] != value["effective"] for value in children):
        raise ValueError(f"Incomplete or missing grouped split for {task}")
    return sum(int(value["effective"]) for value in children)


def metric_value(result: dict[str, Any], task: str, metric: str) -> float:
    for section in ("results", "groups"):
        value = result.get(section, {}).get(task, {}).get(metric)
        if value is not None:
            return float(value)
    raise KeyError(f"Missing {task}.{metric} in lm-eval result")


def validate_existing(path: Path, model: str, task: str, model_path: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    settings = TASKS[task]
    expected = {
        "model": model,
        "task": task,
        "metric": settings["metric"],
        "effective_samples": settings["expected_samples"],
        "num_fewshot": settings["num_fewshot"],
        "batch_size": settings["batch_size"],
        "gen_kwargs": settings["gen_kwargs"],
        "lm_eval_version": LM_EVAL_VERSION,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"Saved result {path} has unexpected {key}; use a fresh output directory")
    if payload.get("model_info", {}).get("model_path") != model_path:
        raise ValueError(f"Saved result {path} came from a different model path")


def write_samples(path: Path, result: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for task_name, samples in result.get("samples", {}).items():
            for sample in samples:
                row = {
                    "task": task_name,
                    "doc_id": sample.get("doc_id"),
                    "target": sample.get("target"),
                    "raw_response": sample.get("resps"),
                    "filtered_response": sample.get("filtered_resps"),
                    "metrics": sample.get("metrics"),
                }
                handle.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")
    temporary.replace(path)


def load_eval_model(model_path: str, batch_size: int):
    """Load a full checkpoint or PEFT adapter with identical lm-eval settings."""
    import torch
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "methods"))
    from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens

    adapter_config_path = Path(model_path) / "adapter_config.json"
    is_adapter = adapter_config_path.is_file()
    base_model = model_path
    tokenizer_source = model_path
    if is_adapter:
        adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model = adapter_config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("adapter_config.json is missing base_model_name_or_path")
        adapter_has_tokenizer = any(
            (Path(model_path) / filename).is_file()
            for filename in ("tokenizer.json", "tokenizer.model", "vocab.json")
        )
        if not adapter_has_tokenizer:
            tokenizer_source = base_model

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, padding_side="left")
    configure_qwen2_base_tokenizer(tokenizer, tokenizer_source)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither PAD nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    use_cuda = torch.cuda.is_available()
    dtype = "bfloat16" if use_cuda and torch.cuda.is_bf16_supported() else ("float16" if use_cuda else "float32")
    model = HFLM(
        pretrained=base_model,
        peft=model_path if is_adapter else None,
        tokenizer=tokenizer,
        dtype=dtype,
        device="cuda" if use_cuda else "cpu",
        parallelize=use_cuda,
        device_map="auto" if use_cuda else None,
        batch_size=batch_size,
    )
    sync_model_special_tokens(model.model, tokenizer)
    inherited_max_new_tokens = getattr(model.model.generation_config, "max_new_tokens", None)
    model.model.generation_config.max_new_tokens = None
    model.model.eval()
    info = {
        "model_path": model_path,
        "model_kind": "peft_adapter" if is_adapter else "full_or_hub_model",
        "base_model": base_model,
        "tokenizer_source": str(tokenizer_source),
        "dtype": dtype,
        "device_map": "auto" if use_cuda else None,
        "padding_side": tokenizer.padding_side,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "disabled_inherited_max_new_tokens": inherited_max_new_tokens,
        "apply_chat_template": bool(tokenizer.chat_template),
    }
    return model, info


def run_worker(args: argparse.Namespace) -> None:
    import lm_eval
    import torch
    from lm_eval.tasks import TaskManager


    model_name = args.worker
    assert model_name is not None
    output = args.output_dir / "models" / model_name
    output.mkdir(parents=True, exist_ok=True)
    pending = []
    for task in TASKS:
        result_path = output / f"{task}.json"
        if result_path.exists() and not args.force:
            validate_existing(result_path, model_name, task, args.model_paths[model_name])
            print(f"[reuse] model={model_name} task={task}", flush=True)
        else:
            pending.append(task)
    if not pending:
        print(f"[reuse] {model_name}: all five tasks complete", flush=True)
        return

    # HFLM stores batch size at construction, so reload when the original
    # benchmark settings switch from GSM8K (batch 1) to the others (batch 4).
    batch_size = TASKS[pending[0]]["batch_size"]
    model, model_info = load_eval_model(args.model_paths[model_name], batch_size)
    task_manager = TaskManager()
    for task in pending:
        settings = TASKS[task]
        if settings["batch_size"] != batch_size:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            batch_size = settings["batch_size"]
            # Keep the configured path when reloading for the next task. This
            # matters for added/overridden methods (and avoids silently
            # evaluating a different checkpoint after the first task).
            model, model_info = load_eval_model(args.model_paths[model_name], batch_size)
        print(f"[run] model={model_name} task={task} batch={batch_size}", flush=True)
        result = lm_eval.simple_evaluate(
            model=model,
            tasks=[task],
            task_manager=task_manager,
            num_fewshot=settings["num_fewshot"],
            batch_size=batch_size,
            apply_chat_template=model_info["apply_chat_template"],
            gen_kwargs=settings["gen_kwargs"],
            limit=None,
            log_samples=True,
            confirm_run_unsafe_code=task == "humaneval_instruct",
            bootstrap_iters=0,
        )
        if result is None:
            raise RuntimeError(f"lm-eval returned no result for {model_name}/{task}")
        count = sample_count(result, task)
        if count != settings["expected_samples"]:
            raise ValueError(f"Unexpected sample count for {task}: {count}")
        score = metric_value(result, task, settings["metric"])
        payload = {
            "model": model_name,
            "model_info": model_info,
            "task": task,
            "metric": settings["metric"],
            "score": score,
            "effective_samples": count,
            "num_fewshot": settings["num_fewshot"],
            "batch_size": batch_size,
            "gen_kwargs": settings["gen_kwargs"],
            "lm_eval_version": LM_EVAL_VERSION,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "lm_eval": {key: value for key, value in result.items() if key != "samples"},
        }
        write_samples(output / f"{task}.samples.jsonl", result)
        atomic_json(output / f"{task}.json", payload)
        print(f"[done] model={model_name} task={task} score={score:.6f}", flush=True)


def write_report(args: argparse.Namespace) -> None:
    rows = []
    for task in TASKS:
        for model in args.models:
            path = args.output_dir / "models" / model / f"{task}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                rows.append({
                    "task": task,
                    "model": model,
                    "metric": payload["metric"],
                    "score": payload["score"],
                    "effective_samples": payload["effective_samples"],
                })
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "scores.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("task", "model", "metric", "score", "effective_samples"))
        writer.writeheader()
        writer.writerows(rows)
    by_pair = {(row["task"], row["model"]): row for row in rows}
    comparison = [model for model in args.models if model != "base"]
    headers = ["Benchmark", *args.models]
    if "base" in args.models:
        headers.extend(f"{model} − Base" for model in comparison)
    lines = [
        "# Stable five benchmark comparison",
        "",
        "Full evaluation splits; scores are percentages; gap columns are percentage points.",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for task in TASKS:
        scores = {
            model: by_pair[(task, model)]["score"] if (task, model) in by_pair else None
            for model in args.models
        }
        cells = [task]
        for model in args.models:
            score = scores[model]
            cells.append(f"{100 * score:.2f}" if score is not None else "pending")
        if "base" in args.models:
            base = scores["base"]
            for model in comparison:
                score = scores[model]
                cells.append(
                    f"{100 * (score - base):+.2f}"
                    if score is not None and base is not None else "pending"
                )
        lines.append("| " + " | ".join(cells) + " |")
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    atomic_json(args.output_dir / "summary.json", {"models": args.model_paths, "tasks": TASKS, "rows": rows})
    print("\n".join(lines), flush=True)


def main() -> None:
    args = arguments()
    installed_version = importlib.metadata.version("lm-eval")
    if installed_version != LM_EVAL_VERSION:
        raise RuntimeError(f"Expected lm-eval {LM_EVAL_VERSION}, found {installed_version}")
    args.output_dir = args.output_dir.resolve()
    if args.worker:
        run_worker(args)
        return
    for model in args.models:
        command = [sys.executable, str(Path(__file__).resolve()), "--worker", model, "--output-dir", str(args.output_dir)]
        for specification in args.model_path:
            command.extend(("--model-path", specification))
        if args.force:
            command.append("--force")
        subprocess.run(command, cwd=ROOT, check=True)
        write_report(args)


if __name__ == "__main__":
    main()
