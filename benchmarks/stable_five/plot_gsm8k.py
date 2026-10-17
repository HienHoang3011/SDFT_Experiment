"""Render Stable Five scores and the separate GSM8K 1024-token rerun."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_scores(path):
    summary = json.loads(path.read_text(encoding="utf-8"))
    return summary, {(row["task"], row["model"]): row["score"] for row in summary["rows"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("rerun", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    previous, old_scores = read_scores(args.previous)
    _, new_scores = read_scores(args.rerun)
    models = list(previous["models"])
    names = {
        "gsm8k_cot_zeroshot": "GSM8K",
        "humaneval_instruct": "HumanEval",
        "ifeval": "IFEval",
        "boolq": "BoolQ",
        "bbh_cot_zeroshot": "BBH",
    }

    def cells(scores, task):
        return [f"{scores[(task, model)] * 100:.2f}%" if (task, model) in scores else "—" for model in models]

    rows = [[names.get(task, task), *cells(old_scores, task)] for task in previous["tasks"]]
    rows.append(["GSM8K · 1024 tokens", *cells(new_scores, "gsm8k_cot_zeroshot")])

    fig, ax = plt.subplots(figsize=(13.5, 5.2))
    ax.axis("off")
    ax.set_title("Stable Five · scores", loc="left", fontsize=20, weight="bold", pad=20)
    table = ax.table(
        cellText=rows,
        colLabels=["Benchmark"] + [model.replace("_", " ").title() for model in models],
        cellLoc="center", loc="center", colWidths=[0.25] + [0.125] * len(models),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dce4ec")
        cell.set_facecolor("#17324d" if row == 0 else ("#e4eff7" if row == len(rows) else ("#f1f5f9" if row % 2 == 0 else "white")))
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row == len(rows):
            cell.get_text().set_weight("bold")
        if cell.get_text().get_text() == "—":
            cell.get_text().set_color("#8a99a8")
        if column == 0:
            cell.get_text().set_ha("left")
    fig.text(0.045, 0.055, "Scores in percent  ·  — = pending  ·  GSM8K 1024 is a separate rerun", fontsize=10, color="#526477")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
