"""Prepare the official SAMSum (EMNLP 2019) splits for supervised summarization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import Dataset, load_dataset

DATASET_ID = "fastlayne/samsum"
SPLITS = ("train", "validation", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="data/samsum_emnlp2019")
    return parser.parse_args()


def prompt_for(dialogue: str) -> str:
    return (
        "Write a concise, faithful summary of the following dialogue.\n\n"
        f"Dialogue:\n{dialogue.strip()}\n\n"
        "Return only the summary inside <answer> and </answer>."
    )


def record(row: dict) -> dict:
    dialogue = str(row["dialogue"]).strip()
    summary = str(row["summary"]).strip()
    if not dialogue or not summary:
        raise ValueError(f"Empty dialogue or summary for SAMSum id={row.get('id')!r}")
    prompt = prompt_for(dialogue)
    return {
        "id": str(row["id"]),
        "dialogue": dialogue,
        "prompt": prompt,
        "summary": summary,
        "output_text": f"<answer>{summary}</answer>",
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<answer>{summary}</answer>"},
        ],
    }


def digest(values: list[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    source = load_dataset(DATASET_ID)
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    prepared = {}
    dropped_empty = {}
    for source_name, destination_name in split_map.items():
        raw_split = source[source_name]
        dataset = raw_split.filter(
            lambda row: bool(str(row["dialogue"]).strip()) and bool(str(row["summary"]).strip()),
            desc=f"Filtering empty {source_name} examples",
        )
        dropped_empty[destination_name] = len(raw_split) - len(dataset)
        if destination_name == "test" and dropped_empty[destination_name]:
            raise AssertionError("Official SAMSum test split contains empty examples")
        prepared[destination_name] = dataset.map(
            record, remove_columns=dataset.column_names
        )

    # SAMSum contains duplicated dialogues under distinct IDs. Evaluation splits
    # take precedence so no dialogue seen in dev/test is present during training.
    dropped_overlap = {}
    protected_dialogues = set()
    for destination_name in ("test", "dev", "train"):
        dataset = prepared[destination_name]
        keep_indices = [
            index
            for index, dialogue in enumerate(dataset["dialogue"])
            if dialogue not in protected_dialogues
        ]
        dropped_overlap[destination_name] = len(dataset) - len(keep_indices)
        prepared[destination_name] = dataset.select(keep_indices)
        protected_dialogues.update(dataset["dialogue"])

    saved = {}
    ids_by_split = {}
    dialogues_by_split = {}
    for destination_name in ("train", "dev", "test"):
        dataset = prepared[destination_name]
        destination = output_root / destination_name
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Refusing to overwrite prepared split: {destination}")
        dataset.save_to_disk(str(destination))
        saved[destination_name] = len(dataset)
        ids_by_split[destination_name] = {str(value) for value in dataset["id"]}
        dialogues_by_split[destination_name] = {str(value) for value in dataset["dialogue"]}

    names = ("train", "dev", "test")
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared_ids = ids_by_split[left] & ids_by_split[right]
            shared_dialogues = dialogues_by_split[left] & dialogues_by_split[right]
            if shared_ids or shared_dialogues:
                raise AssertionError(
                    f"SAMSum split leakage {left}/{right}: "
                    f"ids={len(shared_ids)}, dialogues={len(shared_dialogues)}"
                )
    manifest = {
        "dataset": DATASET_ID,
        "benchmark": "SAMSum (EMNLP 2019)",
        "splits": saved,
        "dropped_empty_examples": dropped_empty,
        "dropped_duplicate_dialogues": dropped_overlap,
        "id_hashes": {name: digest(sorted(values)) for name, values in ids_by_split.items()},
        "dialogue_hashes": {
            name: digest(sorted(values)) for name, values in dialogues_by_split.items()
        },
        "prompt_template": "summary inside <answer> tags",
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
