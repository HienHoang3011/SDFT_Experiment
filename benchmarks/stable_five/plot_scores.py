"""Render benchmark scores from a stable-five summary as a simple image table."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    models = list(summary["models"])
    scores = {(row["task"], row["model"]): row["score"] for row in summary["rows"]}
    tasks = [task for task in summary["tasks"] if any((task, model) in scores for model in models)]
    rows = [
        [task.replace("_", " ").upper() if task.startswith("gsm8k") else task.replace("_", " ").title()]
        + [f"{scores[(task, model)] * 100:.2f}%" if (task, model) in scores else "—" for model in models]
        for task in tasks
    ]

    fig, ax = plt.subplots(figsize=(13, max(2.5, 1.5 + 0.55 * len(tasks))))
    ax.axis("off")
    ax.set_title("Benchmark scores", loc="left", fontsize=19, weight="bold", pad=20)
    table = ax.table(
        cellText=rows,
        colLabels=["Benchmark"] + [model.replace("_", " ").title() for model in models],
        cellLoc="center",
        loc="center",
        colWidths=[0.25] + [0.125] * len(models),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#dce4ec")
        cell.set_facecolor("#17324d" if row == 0 else ("#f1f5f9" if row % 2 == 0 else "white"))
        if row == 0:
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif cell.get_text().get_text() == "—":
            cell.get_text().set_color("#8a99a8")
        if column == 0:
            cell.get_text().set_ha("left")
    fig.text(0.045, 0.07, "Scores in percent  ·  — = pending", fontsize=10, color="#526477")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
