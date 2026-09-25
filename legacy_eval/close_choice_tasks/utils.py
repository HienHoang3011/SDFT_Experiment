"""Prompting, parsing, and alignment helpers for closed-choice generation."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Sequence
from typing import Any


LABELS = ("A", "B", "C", "D")


def _normalize_choice_text(text: str) -> str:
    value = " ".join(unicodedata.normalize("NFKC", str(text)).lower().split())
    start = 0
    end = len(value)
    while start < end and unicodedata.category(value[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(value[end - 1]).startswith("P"):
        end -= 1
    return value[start:end].strip()


def extract_mc_answer(
    text: str, valid_labels: list[str], choices: list[str]
) -> dict[str, str | bool | None]:
    """Extract one answer without treating incidental label letters as answers.

    Models do not always reproduce the requested format literally. In particular,
    instruction-tuned models often render ``<letter>`` from the prompt as an XML
    element, or append the selected choice text after the label. Those are still
    unambiguous answers and should not be counted as parse failures.
    """
    valid = {str(label).upper() for label in valid_labels}
    output = str(text)

    def candidate_label(
        candidate: str, *, allow_trailing_explanation: bool = False
    ) -> str | None:
        letter_tag = re.fullmatch(
            r"\s*<\s*letter\s*>\s*([A-Za-z])\s*"
            r"(?:<\s*/\s*letter\s*>|>)?\s*[.。]?\s*",
            candidate,
            flags=re.IGNORECASE,
        )
        if letter_tag and letter_tag.group(1).upper() in valid:
            return letter_tag.group(1).upper()

        label_match = re.fullmatch(
            r"\s*(?:(?:option|choice)\s+)?(?:\(\s*|\[\s*)?"
            r"([A-Za-z])(?:\s*\)|\s*\])?[.。]?\s*",
            candidate,
            flags=re.IGNORECASE,
        )
        if label_match and label_match.group(1).upper() in valid:
            return label_match.group(1).upper()

        normalized_candidate = _normalize_choice_text(candidate)
        matching_indices = [
            index
            for index, choice in enumerate(choices)
            if _normalize_choice_text(choice) == normalized_candidate
        ]
        if normalized_candidate and len(matching_indices) == 1:
            index = matching_indices[0]
            if index < len(valid_labels):
                return valid_labels[index].upper()

        labelled_choice = re.fullmatch(
            r"\s*(?:(?:option|choice)\s+)?(?:\(\s*|\[\s*)?"
            r"([A-Za-z])(?:\s*\)|\s*\])?"
            r"(?:\s*[.。,:;\-–—]\s*|\s+)(.+?)\s*",
            candidate,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if labelled_choice and labelled_choice.group(1).upper() in valid:
            label = labelled_choice.group(1).upper()
            if allow_trailing_explanation:
                return label
            label_index = valid_labels.index(label)
            if label_index < len(choices) and (
                _normalize_choice_text(labelled_choice.group(2))
                == _normalize_choice_text(choices[label_index])
            ):
                return label
        return None

    def parsed(
        candidate: str,
        method: str,
        *,
        allow_trailing_explanation: bool = False,
    ) -> dict[str, str | bool | None] | None:
        label = candidate_label(
            candidate, allow_trailing_explanation=allow_trailing_explanation
        )
        if label is None:
            return None
        return {
            "predicted_label": label,
            "parse_valid": True,
            "parse_method": method,
        }

    xml_matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for candidate in reversed(xml_matches):
        if result := parsed(
            candidate, "xml_answer", allow_trailing_explanation=True
        ):
            return result

    # The answer is often on the next non-empty line after the marker.
    anchored_candidates: list[str] = []
    lines = output.splitlines()
    answer_marker = re.compile(
        r"(?:^|\b)(?:(?:the\s+)?(?:best|correct|final)\s+answer|"
        r"(?:the\s+)?answer)\s*(?::|=|\bis\b\s*:?)\s*(.*)$",
        flags=re.IGNORECASE,
    )
    for index, line in enumerate(lines):
        marker = answer_marker.search(line)
        if marker is None:
            continue
        candidate = marker.group(1).strip()
        if not candidate:
            candidate = next(
                (later.strip() for later in lines[index + 1 :] if later.strip()),
                "",
            )
        if candidate:
            anchored_candidates.append(candidate)
    for candidate in reversed(anchored_candidates):
        if result := parsed(
            candidate,
            "anchored_final_answer",
            allow_trailing_explanation=True,
        ):
            return result

    letter_payloads = re.findall(
        r"<\s*letter\s*>(.*?)<\s*/\s*letter\s*>",
        output,
        flags=re.IGNORECASE | re.DOTALL,
    )
    for candidate in reversed(letter_payloads):
        candidate = re.sub(
            r"^\s*(?:final\s+answer|answer)\s*(?::|=|\bis\b)?\s*",
            "",
            candidate,
            flags=re.IGNORECASE,
        )
        if result := parsed(candidate, "xml_letter"):
            return result

    letter_tags = re.findall(
        r"<\s*letter\s*>\s*([A-Za-z])\s*(?:<\s*/\s*letter\s*>|>)?",
        output,
        flags=re.IGNORECASE,
    )
    for candidate in reversed(letter_tags):
        if result := parsed(candidate, "xml_letter"):
            return result

    # A label plus its exact corresponding choice is unambiguous even without
    # an explicit "answer" marker.
    for candidate in reversed([line.strip() for line in lines if line.strip()]):
        if result := parsed(candidate, "labelled_choice"):
            return result

    if result := parsed(output, "whole_output_answer"):
        return result

    # Recover concise natural-language answers such as "... is 4" without
    # treating arbitrary choice mentions in reasoning as answers.
    tail = re.search(r"(?:\bis\b|:)\s*([^\n]+?)\s*$", output, re.IGNORECASE)
    if tail and (result := parsed(tail.group(1), "final_choice_text")):
        return result

    return {
        "predicted_label": None,
        "parse_valid": False,
        "parse_method": "invalid",
    }


def format_mc_prompt(question: str, choices: Sequence[str]) -> str:
    if not 2 <= len(choices) <= len(LABELS):
        raise ValueError(f"Expected 2-4 choices, received {len(choices)}")
    rendered_choices = "\n".join(
        f"{LABELS[index]}. {choice}" for index, choice in enumerate(choices)
    )
    allowed_labels = ", ".join(LABELS[: len(choices)])
    return (
        f"Question:\n{question}\n\nChoices:\n{rendered_choices}\n\n"
        f"Select the single best answer. Reply with exactly one of these capital "
        f"letters: {allowed_labels}. Do not include an explanation, punctuation, "
        "or XML tags."
    )


def mmlu_fields(doc: dict[str, Any]) -> tuple[str, list[str], str]:
    choices = [str(choice) for choice in doc["choices"]]
    gold_index = int(doc["answer"])
    return str(doc["question"]).strip(), choices, LABELS[gold_index]


def piqa_fields(doc: dict[str, Any]) -> tuple[str, list[str], str]:
    choices = [str(doc["sol1"]), str(doc["sol2"])]
    gold_index = int(doc["label"])
    return str(doc["goal"]).strip(), choices, LABELS[gold_index]


def boolq_fields(doc: dict[str, Any]) -> tuple[str, list[str], str]:
    question = str(doc["question"]).strip()
    if not question.endswith("?"):
        question += "?"
    combined = f"Passage: {str(doc['passage']).strip()}\nQuestion: {question}"
    choices = ["no", "yes"]
    gold_index = int(doc["label"])
    return combined, choices, LABELS[gold_index]


def mmlu_doc_to_text(doc: dict[str, Any]) -> str:
    question, choices, _ = mmlu_fields(doc)
    return format_mc_prompt(question, choices)


def piqa_doc_to_text(doc: dict[str, Any]) -> str:
    question, choices, _ = piqa_fields(doc)
    return format_mc_prompt(question, choices)


def boolq_doc_to_text(doc: dict[str, Any]) -> str:
    question, choices, _ = boolq_fields(doc)
    return format_mc_prompt(question, choices)


def mmlu_doc_to_target(doc: dict[str, Any]) -> str:
    return mmlu_fields(doc)[2]


def piqa_doc_to_target(doc: dict[str, Any]) -> str:
    return piqa_fields(doc)[2]


def boolq_doc_to_target(doc: dict[str, Any]) -> str:
    return boolq_fields(doc)[2]


def _process_generation(
    doc: dict[str, Any], results: list[str], field_fn
) -> dict[str, float]:
    _, choices, gold_label = field_fn(doc)
    labels = list(LABELS[: len(choices)])
    parsed = extract_mc_answer(str(results[0]), labels, choices)
    return {
        "generation_accuracy": float(parsed["predicted_label"] == gold_label),
        "parse_valid_rate": float(parsed["parse_valid"]),
    }


def process_mmlu_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _process_generation(doc, results, mmlu_fields)


def process_piqa_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _process_generation(doc, results, piqa_fields)


def process_boolq_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    return _process_generation(doc, results, boolq_fields)


def normalize_question(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def sample_fingerprint(
    *,
    dataset_path: str,
    dataset_name: str | None,
    split: str,
    subject: str | None,
    question: str,
    choices: Sequence[str],
    gold_label: str,
) -> str:
    payload = {
        "dataset": str(dataset_path),
        "config": dataset_name,
        "split": str(split),
        "subject": subject,
        "question": normalize_question(question),
        "choices": [normalize_question(choice) for choice in choices],
        "gold_label": str(gold_label),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
