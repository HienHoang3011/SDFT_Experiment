"""Create a compact SAMSum + prior-capabilities report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--full", required=True, type=Path)
    parser.add_argument("--lora", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def row(
    name: str, root: Path
) -> tuple[str, float, float, float, float, float, float, float, float]:
    downstream = load(root / "downstream_results.json")
    prior = load(root / "summary.json")
    closed = prior["closed_log"]
    pure = prior["pure_generation"]
    math500 = pure["math500"]
    humaneval = pure["humaneval"]
    if math500.get("lm_eval_task") != "minerva_math500":
        raise ValueError(f"{root} does not contain standardized MATH-500 results")
    if humaneval.get("lm_eval_task") != "humaneval_instruct":
        raise ValueError(f"{root} does not contain standardized HumanEval results")
    return (
        name,
        100 * float(downstream["rougeL"]),
        100 * float(closed["mmlu"]["canonical_group_score"]),
        100 * float(closed["piqa"]),
        100 * float(closed["boolq"]),
        100 * float(pure["ifeval"]["prompt_level_strict_acc"]),
        100 * float(math500["math_verify"]),
        100 * float(humaneval["pass@1"]),
        100 * float(prior["overall_prior_avg"]),
    )


def save_table_image(
    rows: list[
        tuple[str, float, float, float, float, float, float, float, float]
    ],
    output_dir: Path,
) -> None:
    columns = [
        "Model",
        "SAMSum\nROUGE-L",
        "MMLU\nLog",
        "PIQA\nLog",
        "BoolQ\nLog",
        "IFEval\nStrict",
        "MATH-500\nmath_verify",
        "HumanEval\npass@1",
        "Prior\nAverage",
    ]
    cells = [
        [values[0], *(f"{score:.2f}" for score in values[1:])]
        for values in rows
    ]
    figure, axis = plt.subplots(figsize=(21, 3.0))
    axis.axis("off")
    table = axis.table(
        cellText=cells, colLabels=columns, loc="center", cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.3)
    for (row_index, column_index), cell in table.get_celld().items():
        if row_index == 0:
            cell.set_facecolor("#4c72b0")
            cell.set_text_props(weight="bold", color="white")
        else:
            if column_index in (0, len(columns) - 1):
                cell.set_text_props(weight="bold")
            if row_index % 2 == 0:
                cell.set_facecolor("#f3f4f6")
    figure.patch.set_facecolor("white")
    figure.savefig(
        output_dir / "results_table.png",
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [row("Base", args.base), row("Full SFT", args.full), row("SFT LoRA", args.lora)]
    lines = [
        "# SAMSum (EMNLP 2019) stable experiment",
        "",
        "MMLU/PIQA/BoolQ use canonical log-likelihood only; generated closed-choice answers are disabled.",
        "",
        "| Model | SAMSum ROUGE-L | MMLU log | PIQA log | BoolQ log | IFEval strict | MATH-500 math_verify | HumanEval pass@1 | Prior avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for values in rows:
        lines.append(
            "| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
                *values
            )
        )
    lines += [
        "",
        "Prior avg is the unweighted mean of MMLU log, PIQA log, BoolQ log, IFEval strict, MATH-500 math_verify, and HumanEval pass@1.",
    ]
    (args.output_dir / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_table_image(rows, args.output_dir)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
