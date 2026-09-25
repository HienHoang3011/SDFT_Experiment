"""Evaluate prior capabilities with canonical and generated-answer scoring."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = Path(__file__).resolve().parent
CUSTOM_TASK_DIR = EVAL_DIR / "close_choice_tasks"
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(ROOT / "methods"))

from close_choice_tasks.utils import (  # noqa: E402
    LABELS,
    boolq_fields,
    extract_mc_answer,
    mmlu_fields,
    piqa_fields,
    sample_fingerprint,
)
from qwen2_base import (  # noqa: E402
    configure_qwen2_base_tokenizer,
    sync_model_special_tokens,
)


os.environ["HF_ALLOW_CODE_EVAL"] = "1"

CANONICAL_TASKS = ("mmlu", "piqa", "boolq", "ifeval")
CLOSED_GENERATION_TASKS = ("mmlu_gen", "piqa_gen", "boolq_gen")
STANDARD_GENERATION_TASKS = ("minerva_math500", "humaneval_instruct")
EXPECTED_STANDARD_SAMPLE_COUNTS = {
    "minerva_math500": 500,
    "humaneval_instruct": 164,
}
PRIMARY_METRICS = {
    "mmlu": ("groups", "mmlu", "acc,none"),
    "piqa": ("results", "piqa", "acc_norm,none"),
    "boolq": ("results", "boolq", "acc,none"),
    "ifeval_prompt_strict": ("results", "ifeval", "prompt_level_strict_acc,none"),
    "ifeval_prompt_loose": ("results", "ifeval", "prompt_level_loose_acc,none"),
    "ifeval_inst_strict": ("results", "ifeval", "inst_level_strict_acc,none"),
    "ifeval_inst_loose": ("results", "ifeval", "inst_level_loose_acc,none"),
    "math500": ("results", "minerva_math500", "math_verify,none"),
    "humaneval": (
        "results",
        "humaneval_instruct",
        "pass@1,create_test",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete six-benchmark prior-capabilities evaluation."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--include_closed_generation",
        action="store_true",
        help=(
            "Also generate and parse answers for MMLU/PIQA/BoolQ. Disabled by "
            "default: headline closed-choice scores use log-likelihood only."
        ),
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be a positive integer")
    return args


def _mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"Cannot average invalid metrics: {values}")
    return sum(values) / len(values)


def _metric(result: dict[str, Any], key: str) -> float:
    section, task_name, metric_name = PRIMARY_METRICS[key]
    try:
        value = result[section][task_name][metric_name]
    except KeyError as exc:
        raise KeyError(
            f"lm-eval 0.4.12 did not return required metric "
            f"{section}.{task_name}.{metric_name}"
        ) from exc
    if not isinstance(value, (int, float)):
        raise TypeError(f"Metric {task_name}.{metric_name} is not numeric: {value!r}")
    return float(value)


def _task_metric(
    result: dict[str, Any], task_name: str, metric_name: str
) -> float:
    try:
        value = result["results"][task_name][metric_name]
    except KeyError as exc:
        raise KeyError(f"Missing exact metric {task_name}.{metric_name}") from exc
    if not isinstance(value, (int, float)):
        raise TypeError(f"Metric {task_name}.{metric_name} is not numeric")
    return float(value)


def _sample_count(result: dict[str, Any], task_name: str) -> int:
    try:
        counts = result["n-samples"][task_name]
        original = int(counts["original"])
        effective = int(counts["effective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing sample count for {task_name}") from exc
    if original != effective:
        raise AssertionError(
            f"{task_name} did not run its full evaluation split: "
            f"original={original}, effective={effective}"
        )
    return effective


def _config_value(task: Any, name: str) -> Any:
    return getattr(task.config, name)


def _task_fingerprints(
    task: Any, benchmark: str, subject: str | None
) -> list[str]:
    field_fn = {"mmlu": mmlu_fields, "piqa": piqa_fields, "boolq": boolq_fields}[
        benchmark
    ]
    split = _config_value(task, "test_split") or _config_value(
        task, "validation_split"
    )
    fingerprints = []
    for doc in task.eval_docs:
        question, choices, gold_label = field_fn(doc)
        fingerprints.append(
            sample_fingerprint(
                dataset_path=_config_value(task, "dataset_path"),
                dataset_name=_config_value(task, "dataset_name"),
                split=split,
                subject=subject,
                question=question,
                choices=choices,
                gold_label=gold_label,
            )
        )
    return fingerprints


def build_alignment(task_manager: Any) -> dict[str, Any]:
    canonical = task_manager.load(["mmlu", "piqa", "boolq"])["tasks"]
    generated = task_manager.load(list(CLOSED_GENERATION_TASKS))["tasks"]
    details: dict[str, Any] = {"mmlu": {"subjects": {}}, "piqa": {}, "boolq": {}}

    canonical_subjects = sorted(
        name.removeprefix("mmlu_")
        for name in canonical
        if name.startswith("mmlu_")
    )
    generated_subjects = sorted(
        name.removeprefix("mmlu_gen_")
        for name in generated
        if name.startswith("mmlu_gen_")
    )
    if canonical_subjects != generated_subjects:
        raise AssertionError("mmlu_gen subjects differ from canonical mmlu subjects")

    mmlu_log_total = 0
    mmlu_gen_total = 0
    for subject in canonical_subjects:
        log_name = f"mmlu_{subject}"
        gen_name = f"mmlu_gen_{subject}"
        log_hashes = _task_fingerprints(canonical[log_name], "mmlu", subject)
        gen_hashes = _task_fingerprints(generated[gen_name], "mmlu", subject)
        if log_hashes != gen_hashes:
            raise AssertionError(f"MMLU fingerprint mismatch for subject {subject}")
        mmlu_log_total += len(log_hashes)
        mmlu_gen_total += len(gen_hashes)
        details["mmlu"]["subjects"][subject] = {
            "aligned": True,
            "log_count": len(log_hashes),
            "generation_count": len(gen_hashes),
            "log_fingerprints": log_hashes,
            "generation_fingerprints": gen_hashes,
        }
    if mmlu_log_total != mmlu_gen_total:
        raise AssertionError("MMLU total sample counts differ")
    details["mmlu"].update(
        {
            "aligned": True,
            "log_total_samples": mmlu_log_total,
            "generation_total_samples": mmlu_gen_total,
        }
    )

    for benchmark in ("piqa", "boolq"):
        log_hashes = _task_fingerprints(canonical[benchmark], benchmark, None)
        gen_hashes = _task_fingerprints(generated[f"{benchmark}_gen"], benchmark, None)
        if log_hashes != gen_hashes:
            raise AssertionError(f"{benchmark} fingerprints do not match")
        details[benchmark] = {
            "aligned": True,
            "log_count": len(log_hashes),
            "generation_count": len(gen_hashes),
            "log_fingerprints": log_hashes,
            "generation_fingerprints": gen_hashes,
        }
    details["all_aligned"] = True
    return details


def load_eval_model(model_path: str, batch_size: int):
    import torch
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoTokenizer

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
    if use_cuda:
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        dtype = "float32"
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


def build_standard_generation_samples(
    standard_result: dict[str, Any],
) -> list[dict[str, Any]]:
    """Keep auditable raw responses for standardized generation benchmarks."""
    rows: list[dict[str, Any]] = []
    task_aliases = {
        "minerva_math500": "math500",
        "humaneval_instruct": "humaneval",
    }
    for task_name in STANDARD_GENERATION_TASKS:
        for sample in standard_result["samples"][task_name]:
            raw_container = sample.get("resps", sample.get("filtered_resps"))
            filtered_container = sample.get("filtered_resps")
            rows.append(
                {
                    "task": task_aliases[task_name],
                    "lm_eval_task": task_name,
                    "doc_index": int(sample["doc_id"]),
                    "doc": sample.get("doc"),
                    "raw_response": _unwrap_response(raw_container),
                    "filtered_response": (
                        _unwrap_response(filtered_container)
                        if filtered_container is not None
                        else None
                    ),
                    "metrics": sample.get("metrics"),
                }
            )
    return rows


def _unwrap_response(value: Any) -> str:
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return str(value)


def build_closed_generation_samples(
    generation_result: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = []
    for task_name in sorted(
        name for name in generation_result["samples"] if name.startswith("mmlu_gen_")
    ):
        specs.append((task_name, "mmlu", task_name.removeprefix("mmlu_gen_"), mmlu_fields))
    specs.extend(
        [
            ("piqa_gen", "piqa", None, piqa_fields),
            ("boolq_gen", "boolq", None, boolq_fields),
        ]
    )
    for task_name, benchmark, subject, field_fn in specs:
        for sample in generation_result["samples"][task_name]:
            doc = sample["doc"]
            question, choices, gold_label = field_fn(doc)
            raw_container = sample.get("resps", sample.get("filtered_resps"))
            raw_response = _unwrap_response(raw_container)
            parsed = extract_mc_answer(
                raw_response, list(LABELS[: len(choices)]), choices
            )
            config = generation_result["configs"][task_name]
            split = config.get("test_split") or config.get("validation_split")
            rows.append(
                {
                    "task": benchmark,
                    "subject": subject,
                    "doc_index": int(sample["doc_id"]),
                    "fingerprint": sample_fingerprint(
                        dataset_path=config["dataset_path"],
                        dataset_name=config.get("dataset_name"),
                        split=split,
                        subject=subject,
                        question=question,
                        choices=choices,
                        gold_label=gold_label,
                    ),
                    "question": question,
                    "choices": choices,
                    "gold_label": gold_label,
                    "raw_response": raw_response,
                    **parsed,
                    "correct": parsed["predicted_label"] == gold_label,
                }
            )
    return rows


def validate_result_counts(
    canonical_result: dict[str, Any], generation_result: dict[str, Any]
) -> None:
    for benchmark in ("piqa", "boolq"):
        log_count = _sample_count(canonical_result, benchmark)
        gen_count = _sample_count(generation_result, f"{benchmark}_gen")
        if log_count != gen_count:
            raise AssertionError(f"{benchmark} evaluated sample counts differ")

    log_subjects = sorted(
        name for name in canonical_result["n-samples"] if name.startswith("mmlu_")
    )
    for log_name in log_subjects:
        subject = log_name.removeprefix("mmlu_")
        gen_name = f"mmlu_gen_{subject}"
        if _sample_count(canonical_result, log_name) != _sample_count(
            generation_result, gen_name
        ):
            raise AssertionError(f"MMLU evaluated counts differ for {subject}")
    log_total = sum(_sample_count(canonical_result, name) for name in log_subjects)
    gen_total = sum(
        _sample_count(generation_result, f"mmlu_gen_{name.removeprefix('mmlu_')}")
        for name in log_subjects
    )
    if log_total != gen_total:
        raise AssertionError("MMLU log/generation total evaluated counts differ")


def build_summary(
    canonical_result: dict[str, Any],
    generation_result: dict[str, Any] | None,
    standard_result: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    mmlu = _metric(canonical_result, "mmlu")
    piqa = _metric(canonical_result, "piqa")
    boolq = _metric(canonical_result, "boolq")
    ifeval_strict = _metric(canonical_result, "ifeval_prompt_strict")
    math500_score = _metric(standard_result, "math500")
    humaneval_pass1 = _metric(standard_result, "humaneval")
    if generation_result is None:
        closed_generation = {"enabled": False}
    else:
        mmlu_gen = float(
            generation_result["groups"]["mmlu_gen"]["generation_accuracy,none"]
        )
        closed_generation = {
            "enabled": True,
            "mmlu": {
                "accuracy": mmlu_gen,
                "parse_valid_rate": float(
                    generation_result["groups"]["mmlu_gen"]["parse_valid_rate,none"]
                ),
            },
            "piqa": {
                "accuracy": _task_metric(
                    generation_result, "piqa_gen", "generation_accuracy,none"
                ),
                "parse_valid_rate": _task_metric(
                    generation_result, "piqa_gen", "parse_valid_rate,none"
                ),
            },
            "boolq": {
                "accuracy": _task_metric(
                    generation_result, "boolq_gen", "generation_accuracy,none"
                ),
                "parse_valid_rate": _task_metric(
                    generation_result, "boolq_gen", "parse_valid_rate,none"
                ),
            },
        }
        closed_generation["average"] = _mean(
            [
                mmlu_gen,
                closed_generation["piqa"]["accuracy"],
                closed_generation["boolq"]["accuracy"],
            ]
        )
    pure_generation = {
        "math500": {
            "math_verify": math500_score,
            "num_samples": _sample_count(standard_result, "minerva_math500"),
            "lm_eval_task": "minerva_math500",
        },
        "humaneval": {
            "pass@1": humaneval_pass1,
            "num_samples": _sample_count(standard_result, "humaneval_instruct"),
            "lm_eval_task": "humaneval_instruct",
        },
        "ifeval": {
            "prompt_level_strict_acc": ifeval_strict,
            "prompt_level_loose_acc": _metric(canonical_result, "ifeval_prompt_loose"),
            "inst_level_strict_acc": _metric(canonical_result, "ifeval_inst_strict"),
            "inst_level_loose_acc": _metric(canonical_result, "ifeval_inst_loose"),
        },
    }
    pure_generation["average"] = _mean(
        [
            math500_score,
            humaneval_pass1,
            ifeval_strict,
        ]
    )
    return {
        "sample_alignment": {
            "mmlu": bool(alignment["mmlu"]["aligned"]),
            "piqa": bool(alignment["piqa"]["aligned"]),
            "boolq": bool(alignment["boolq"]["aligned"]),
            "all_aligned": bool(alignment["all_aligned"]),
        },
        "closed_log": {
            "mmlu": {"canonical_group_score": mmlu},
            "piqa": piqa,
            "boolq": boolq,
            "average": _mean([mmlu, piqa, boolq]),
        },
        "closed_generation": closed_generation,
        "pure_generation": pure_generation,
        "overall_prior_avg": _mean(
            [
                mmlu,
                piqa,
                boolq,
                math500_score,
                humaneval_pass1,
                ifeval_strict,
            ]
        ),
    }


def _json_default(value: Any) -> Any:
    if callable(value):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    import lm_eval
    from lm_eval.tasks import TaskManager

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_manager = TaskManager(include_path=str(CUSTOM_TASK_DIR))

    print("[1/6] Verifying closed-choice full-document alignment...")
    alignment = build_alignment(task_manager)
    print("[2/6] Loading model/tokenizer...")
    model, model_info = load_eval_model(args.model_path, args.batch_size)
    apply_chat_template = model_info["apply_chat_template"]

    print("[3/6] Running canonical mmlu/piqa/boolq/ifeval...")
    canonical_result = lm_eval.simple_evaluate(
        model=model,
        tasks=list(CANONICAL_TASKS),
        task_manager=task_manager,
        batch_size=args.batch_size,
        apply_chat_template=apply_chat_template,
        log_samples=False,
    )
    if canonical_result is None:
        raise RuntimeError("lm-eval returned no canonical results")

    generation_result = None
    closed_rows = []
    if args.include_closed_generation:
        print("[4/6] Running generated-answer audit for mmlu/piqa/boolq...")
        generation_result = lm_eval.simple_evaluate(
            model=model,
            tasks=list(CLOSED_GENERATION_TASKS),
            task_manager=task_manager,
            batch_size=args.batch_size,
            apply_chat_template=apply_chat_template,
            log_samples=True,
        )
        if generation_result is None:
            raise RuntimeError("lm-eval returned no closed-generation results")
        validate_result_counts(canonical_result, generation_result)
        closed_rows = build_closed_generation_samples(generation_result)
        expected_rows = sum(
            _sample_count(generation_result, task_name)
            for task_name in generation_result["samples"]
        )
        if len(closed_rows) != expected_rows:
            raise AssertionError(
                "Closed-generation JSONL count mismatch: "
                f"{len(closed_rows)} != {expected_rows}"
            )
    else:
        print("[4/6] Skipping closed-choice generation audit (log-only default).")

    print("[5/6] Running standardized MATH-500 and HumanEval-Instruct...")
    standard_result = lm_eval.simple_evaluate(
        model=model,
        tasks=list(STANDARD_GENERATION_TASKS),
        task_manager=task_manager,
        batch_size=args.batch_size,
        apply_chat_template=apply_chat_template,
        log_samples=True,
        confirm_run_unsafe_code=True,
    )
    if standard_result is None:
        raise RuntimeError("lm-eval returned no standardized generation results")
    for task_name, expected_count in EXPECTED_STANDARD_SAMPLE_COUNTS.items():
        actual_count = _sample_count(standard_result, task_name)
        if actual_count != expected_count:
            raise AssertionError(
                f"{task_name} sample count changed: "
                f"expected={expected_count}, actual={actual_count}"
            )
    standard_rows = build_standard_generation_samples(standard_result)
    expected_standard_rows = sum(
        _sample_count(standard_result, task_name)
        for task_name in STANDARD_GENERATION_TASKS
    )
    if len(standard_rows) != expected_standard_rows:
        raise AssertionError(
            "Standard-generation JSONL count mismatch: "
            f"{len(standard_rows)} != {expected_standard_rows}"
        )

    print("[6/6] Building standardized summary and auditable artifacts...")
    summary = build_summary(
        canonical_result, generation_result, standard_result, alignment
    )

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "sample_alignment.json", alignment)
    _write_json(
        output_dir / "detailed_lm_eval.json",
        {
            "canonical": {k: v for k, v in canonical_result.items() if k != "samples"},
            "closed_generation": (
                {k: v for k, v in generation_result.items() if k != "samples"}
                if generation_result is not None
                else {"enabled": False}
            ),
            "standard_generation": {
                k: v for k, v in standard_result.items() if k != "samples"
            },
            "primary_metric_keys": PRIMARY_METRICS,
        },
    )
    if generation_result is not None:
        with (output_dir / "closed_generation_samples.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in closed_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "generation_only_samples.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in standard_rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, default=_json_default)
            )
            handle.write("\n")
    _write_json(
        output_dir / "run_config.json",
        {
            **model_info,
            "batch_size": args.batch_size,
            "lm_eval_version": importlib.metadata.version("lm-eval"),
            "canonical_lm_eval_tasks": list(CANONICAL_TASKS),
            "closed_generation_enabled": args.include_closed_generation,
            "closed_generation_lm_eval_tasks": (
                list(CLOSED_GENERATION_TASKS)
                if args.include_closed_generation
                else []
            ),
            "standard_generation_lm_eval_tasks": list(
                STANDARD_GENERATION_TASKS
            ),
            "standard_generation_log_samples": True,
            "full_evaluation": True,
            "limit": None,
            "closed_generation_kwargs": {
                "output_type": "generate_until",
                "do_sample": False,
                "temperature": 0.0,
                "max_gen_toks": 32,
                "eos_token_id": model_info["eos_token_id"],
            },
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Results written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
