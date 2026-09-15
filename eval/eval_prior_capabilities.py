"""Evaluate prior capabilities with canonical and generated-answer scoring."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import re
import sys
import unicodedata
from datetime import datetime, timezone
from fractions import Fraction
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


import os
os.environ["HF_ALLOW_CODE_EVAL"] = "1"

CANONICAL_TASKS = ("mmlu", "piqa", "boolq", "ifeval", "humaneval")
CLOSED_GENERATION_TASKS = ("mmlu_gen", "piqa_gen", "boolq_gen")
EXTERNAL_GENERATION_TASKS = ("math500",)
PRIMARY_METRICS = {
    "mmlu": ("groups", "mmlu", "acc,none"),
    "piqa": ("results", "piqa", "acc_norm,none"),
    "boolq": ("results", "boolq", "acc,none"),
    "ifeval_prompt_strict": ("results", "ifeval", "prompt_level_strict_acc,none"),
    "ifeval_prompt_loose": ("results", "ifeval", "prompt_level_loose_acc,none"),
    "ifeval_inst_strict": ("results", "ifeval", "inst_level_strict_acc,none"),
    "ifeval_inst_loose": ("results", "ifeval", "inst_level_loose_acc,none"),
    "humaneval": ("results", "humaneval", "pass@1,create_test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete six-benchmark prior-capabilities evaluation."
    )
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
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


def _strip_balanced_box(text: str) -> str | None:
    """Return the last balanced LaTeX boxed/fbox payload."""
    candidates: list[str] = []
    for match in re.finditer(r"\\(?:boxed|fbox)\s*\{", text):
        start = match.end()
        depth = 1
        for index in range(start, len(text)):
            if text[index] == "{":
                depth += 1
            elif text[index] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:index].strip())
                    break
    return candidates[-1] if candidates else None


def extract_freeform_answer(text: str) -> dict[str, str | bool | None]:
    """Extract strict and flexible final answers from a generated response."""
    output = str(text).strip()
    strict_matches = re.findall(
        r"(?:^|\n)\s*Final\s+answer\s*:\s*([^\n]+?)\s*$",
        output,
        flags=re.IGNORECASE,
    )
    strict_answer = strict_matches[-1].strip() if strict_matches else None
    if strict_answer:
        return {
            "strict_answer": strict_answer,
            "strict_parse_valid": True,
            "flexible_answer": strict_answer,
            "flexible_parse_valid": True,
            "parse_method": "final_answer",
        }

    xml_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>", output, flags=re.IGNORECASE | re.DOTALL
    )
    if xml_matches and xml_matches[-1].strip():
        return {
            "strict_answer": None,
            "strict_parse_valid": False,
            "flexible_answer": xml_matches[-1].strip(),
            "flexible_parse_valid": True,
            "parse_method": "xml_answer",
        }

    boxed = _strip_balanced_box(output)
    if boxed:
        return {
            "strict_answer": None,
            "strict_parse_valid": False,
            "flexible_answer": boxed,
            "flexible_parse_valid": True,
            "parse_method": "latex_boxed",
        }

    anchored_matches = re.findall(
        r"(?:^|\n)\s*(?:(?:the\s+)?(?:final\s+|correct\s+)?answer\s+is\s*:?|"
        r"answer\s*:)\s*([^\n]+?)\s*$",
        output,
        flags=re.IGNORECASE,
    )
    if anchored_matches and anchored_matches[-1].strip():
        return {
            "strict_answer": None,
            "strict_parse_valid": False,
            "flexible_answer": anchored_matches[-1].strip(),
            "flexible_parse_valid": True,
            "parse_method": "anchored_answer",
        }

    hash_matches = re.findall(r"####\s*([^\n]+)", output)
    if hash_matches and hash_matches[-1].strip():
        return {
            "strict_answer": None,
            "strict_parse_valid": False,
            "flexible_answer": hash_matches[-1].strip(),
            "flexible_parse_valid": True,
            "parse_method": "hash_answer",
        }

    nonempty_lines = [line.strip() for line in output.splitlines() if line.strip()]
    if nonempty_lines:
        last_line = re.sub(
            r"^(?:therefore,?\s*)?(?:the\s+)?answer\s*(?:is|:)?\s*",
            "",
            nonempty_lines[-1],
            flags=re.IGNORECASE,
        ).strip()
        if last_line:
            return {
                "strict_answer": None,
                "strict_parse_valid": False,
                "flexible_answer": last_line,
                "flexible_parse_valid": True,
                "parse_method": "last_nonempty_line",
            }

    return {
        "strict_answer": None,
        "strict_parse_valid": False,
        "flexible_answer": None,
        "flexible_parse_valid": False,
        "parse_method": "invalid",
    }


def _normalize_math_answer(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).strip()
    boxed = _strip_balanced_box(value)
    if boxed is not None:
        value = boxed
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("\\(", "").replace("\\)", "")
    value = value.strip("$ ")
    value = re.sub(r"\\text\{([^{}]*)\}", r"\1", value)
    value = re.sub(r"\s+", "", value)
    return value.rstrip(".,;").casefold()


def _as_fraction(text: str) -> Fraction | None:
    value = _normalize_math_answer(text).replace(",", "")
    latex_fraction = re.fullmatch(
        r"([+-]?)\\frac\{([+-]?\d+)\}\{([+-]?\d+)\}", value
    )
    try:
        if latex_fraction:
            sign, numerator, denominator = latex_fraction.groups()
            result = Fraction(int(numerator), int(denominator))
            return -result if sign == "-" else result
        if re.fullmatch(r"[+-]?\d+/[+-]?\d+", value):
            numerator, denominator = value.split("/", 1)
            return Fraction(int(numerator), int(denominator))
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:%|\\%)?", value):
            percent = value.endswith("%") or value.endswith("\\%")
            numeric = value.removesuffix("\\%").removesuffix("%")
            result = Fraction(numeric)
            return result / 100 if percent else result
    except (ValueError, ZeroDivisionError):
        return None
    return None


def math_answer_equal(prediction: str | None, gold: str) -> bool:
    if not prediction:
        return False
    if _normalize_math_answer(prediction) == _normalize_math_answer(gold):
        return True
    predicted_number = _as_fraction(prediction)
    gold_number = _as_fraction(gold)
    return (
        predicted_number is not None
        and gold_number is not None
        and predicted_number == gold_number
    )


def _normalize_factual_answer(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def factual_answer_equal(
    prediction: str | None, gold: str, *, allow_containment: bool
) -> bool:
    if not prediction:
        return False
    predicted = _normalize_factual_answer(prediction)
    target = _normalize_factual_answer(gold)
    if not predicted or not target:
        return False
    if predicted == target:
        return True
    return allow_containment and f" {target} " in f" {predicted} "


def _render_generation_prompt(task: str, question: str) -> str:
    instruction = (
        "Solve the mathematics problem. You may show concise reasoning."
        if task == "math500"
        else "Answer the factual question concisely."
    )
    return (
        f"{instruction}\n\nQuestion:\n{question}\n\n"
        "End your response with exactly:\nFinal answer: <answer>"
    )


def _load_external_generation_datasets():
    from datasets import load_dataset

    return {
        "math500": load_dataset("HuggingFaceH4/MATH-500", split="test"),
    }


def evaluate_external_generation(model: Any, batch_size: int):
    """Run the full MATH-500 and SimpleQA splits without lm-eval."""
    import torch

    tokenizer = model.tokenizer
    datasets_by_task = _load_external_generation_datasets()
    rows: list[dict[str, Any]] = []
    metrics: dict[str, dict[str, float | int]] = {}
    max_new_tokens = {"math500": 512}

    for task in EXTERNAL_GENERATION_TASKS:
        dataset = datasets_by_task[task]
        strict_correct = 0
        flexible_correct = 0
        parse_valid = 0
        for start in range(0, len(dataset), batch_size):
            batch = dataset[start : start + batch_size]
            prompts = [
                _render_generation_prompt(task, question)
                for question in batch["problem"]
            ]
            if tokenizer.chat_template:
                prompts = [
                    tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for prompt in prompts
                ]
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True
            )
            inputs = {
                key: value.to(model.device) for key, value in inputs.items()
            }
            with torch.inference_mode():
                generated = model.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens[task],
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            responses = tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for offset, raw_response in enumerate(responses):
                index = start + offset
                gold = str(batch["answer"][offset])
                parsed = extract_freeform_answer(raw_response)
                if task == "math500":
                    strict_ok = math_answer_equal(parsed["strict_answer"], gold)
                    flexible_ok = math_answer_equal(parsed["flexible_answer"], gold)
                    sample_id = str(batch["unique_id"][offset])
                else:
                    strict_ok = factual_answer_equal(
                        parsed["strict_answer"], gold, allow_containment=False
                    )
                    flexible_ok = factual_answer_equal(
                        parsed["flexible_answer"], gold, allow_containment=True
                    )
                    sample_id = str(index)
                strict_correct += int(strict_ok)
                flexible_correct += int(flexible_ok)
                parse_valid += int(parsed["flexible_parse_valid"])
                rows.append(
                    {
                        "task": task,
                        "sample_id": sample_id,
                        "doc_index": index,
                        "question": batch["problem"][offset],
                        "gold_answer": gold,
                        "raw_response": raw_response,
                        **parsed,
                        "strict_correct": strict_ok,
                        "flexible_correct": flexible_ok,
                    }
                )
            print(f"[{task}] {min(start + batch_size, len(dataset))}/{len(dataset)}")
        count = len(dataset)
        metrics[task] = {
            "strict_match": strict_correct / count,
            "flexible_match": flexible_correct / count,
            "parse_valid_rate": parse_valid / count,
            "num_samples": count,
        }
    return metrics, rows


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
    generation_result: dict[str, Any],
    external_metrics: dict[str, dict[str, float | int]],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    mmlu = _metric(canonical_result, "mmlu")
    piqa = _metric(canonical_result, "piqa")
    boolq = _metric(canonical_result, "boolq")
    ifeval_strict = _metric(canonical_result, "ifeval_prompt_strict")
    humaneval_pass1 = _metric(canonical_result, "humaneval")
    mmlu_gen = float(
        generation_result["groups"]["mmlu_gen"]["generation_accuracy,none"]
    )
    closed_generation = {
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
        "math500": external_metrics["math500"],
        "humaneval": {"pass@1": humaneval_pass1},
        "ifeval": {
            "prompt_level_strict_acc": ifeval_strict,
            "prompt_level_loose_acc": _metric(canonical_result, "ifeval_prompt_loose"),
            "inst_level_strict_acc": _metric(canonical_result, "ifeval_inst_strict"),
            "inst_level_loose_acc": _metric(canonical_result, "ifeval_inst_loose"),
        },
    }
    pure_generation["average"] = _mean(
        [
            float(external_metrics["math500"]["flexible_match"]),
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
                float(external_metrics["math500"]["flexible_match"]),
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

    print("[1/5] Verifying closed-choice full-document alignment...")
    alignment = build_alignment(task_manager)
    print("[2/5] Loading model/tokenizer...")
    model, model_info = load_eval_model(args.model_path, args.batch_size)
    apply_chat_template = model_info["apply_chat_template"]

    print("[3/5] Running canonical mmlu/piqa/boolq/ifeval...")
    canonical_result = lm_eval.simple_evaluate(
        model=model,
        tasks=list(CANONICAL_TASKS),
        task_manager=task_manager,
        batch_size=args.batch_size,
        apply_chat_template=apply_chat_template,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )
    if canonical_result is None:
        raise RuntimeError("lm-eval returned no canonical results")

    print("[4/5] Running mmlu_gen/piqa_gen/boolq_gen through lm-eval...")
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
            f"Closed-generation JSONL count mismatch: {len(closed_rows)} != {expected_rows}"
        )

    print("[5/5] Running full MATH-500 and SimpleQA generation without lm-eval...")
    external_metrics, external_rows = evaluate_external_generation(
        model, args.batch_size
    )
    summary = build_summary(
        canonical_result, generation_result, external_metrics, alignment
    )

    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "sample_alignment.json", alignment)
    _write_json(
        output_dir / "detailed_lm_eval.json",
        {
            "canonical": {k: v for k, v in canonical_result.items() if k != "samples"},
            "closed_generation": {
                k: v for k, v in generation_result.items() if k != "samples"
            },
            "external_generation": external_metrics,
            "primary_metric_keys": PRIMARY_METRICS,
        },
    )
    with (output_dir / "closed_generation_samples.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in closed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "generation_only_samples.jsonl").open(
        "w", encoding="utf-8"
    ) as handle:
        for row in external_rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")
    _write_json(
        output_dir / "run_config.json",
        {
            **model_info,
            "batch_size": args.batch_size,
            "lm_eval_version": importlib.metadata.version("lm-eval"),
            "canonical_lm_eval_tasks": list(CANONICAL_TASKS),
            "closed_generation_lm_eval_tasks": list(CLOSED_GENERATION_TASKS),
            "external_generation_tasks": {
                "math500": {
                    "dataset": "HuggingFaceH4/MATH-500",
                    "config": "default",
                    "split": "test",
                    "max_new_tokens": 512,
                }
            },
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
