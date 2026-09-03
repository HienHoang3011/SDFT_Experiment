"""Generate old-style tables and plots for one CF screening task."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ("base", "full", "lora")
METHOD_LABELS = {"base": "Base Model", "full": "Full SFT", "lora": "SFT LoRA"}
COLORS = {"base": "#999999", "full": "#d62728", "lora": "#ff7f0e"}
# One representative score per prior benchmark. WinoGrande is a downstream
# task in this experiment, so it must not contribute to retained capability.
PRIOR_TASKS = (
    ("hellaswag", "HellaSwag"),
    ("mmlu", "MMLU"),
    ("truthfulqa_mc2", "TruthfulQA"),
    ("humaneval", "HumanEval"),
    ("ifeval", "IFEval"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--report_dir", required=True)
    return parser.parse_args()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def numeric_value(mapping, key):
    value = mapping.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def mean_available(values):
    available = [value for value in values if value is not None]
    return sum(available) / len(available) if available else None


def format_percent(value, signed=False):
    if value is None:
        return "N/A"
    return f"{value * 100:+.2f}" if signed else f"{value * 100:.2f}"


def save_markdown(report_dir, task, metric, downstream, priors, averages, summary):
    headers = [
        "Method",
        *[label for _, label in PRIOR_TASKS],
        "Prior Avg",
        f"{task} {metric}",
    ]
    lines = [
        f"# CF screening report: {task}",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for method in METHODS:
        cells = [METHOD_LABELS[method]]
        cells.extend(format_percent(priors[method][key]) for key, _ in PRIOR_TASKS)
        cells.extend([format_percent(averages[method]), format_percent(downstream[method])])
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Delta versus Base",
        "",
        "| Method | Downstream delta | Prior-average delta | MMLU delta |",
        "|---|---:|---:|---:|",
    ]
    for method in ("full", "lora"):
        lines.append(
            f"| {METHOD_LABELS[method]} | "
            f"{format_percent(summary['downstream_delta_vs_base'][method], True)} | "
            f"{format_percent(summary['prior_average_delta_vs_base'][method], True)} | "
            f"{format_percent(summary['mmlu_delta_vs_base'][method], True)} |"
        )
    lines += ["", "## Criteria", ""]
    lines += [f"- {name}: **{value}**" for name, value in summary["criteria"].items()]
    (report_dir / "results_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_image_table(report_dir, task, metric, downstream, priors, averages):
    headers = [
        "Method",
        *[label for _, label in PRIOR_TASKS],
        "Prior Avg",
        f"{task}\n{metric}",
    ]
    rows = []
    for method in METHODS:
        row = [METHOD_LABELS[method]]
        row.extend(format_percent(priors[method][key]) for key, _ in PRIOR_TASKS)
        row.extend([format_percent(averages[method]), format_percent(downstream[method])])
        rows.append(row)

    figure, axis = plt.subplots(figsize=(14, 1 + 0.7 * len(rows)))
    axis.axis("off")
    table = axis.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    for (row, column), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#4c72b0")
        elif column in (0, len(headers) - 2, len(headers) - 1):
            cell.set_text_props(weight="bold")
        if row > 0 and row % 2 == 0:
            cell.set_facecolor("#f3f4f6")
    figure.savefig(report_dir / "results_table.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_tradeoff_plot(report_dir, task, metric, downstream, averages):
    figure, axis = plt.subplots(figsize=(9, 6))
    base_x = downstream["base"] * 100
    base_y = None if averages["base"] is None else averages["base"] * 100
    for method in METHODS:
        if averages[method] is None:
            continue
        x_value = downstream[method] * 100
        y_value = averages[method] * 100
        axis.scatter(x_value, y_value, color=COLORS[method], s=90, zorder=5)
        axis.annotate(
            METHOD_LABELS[method],
            (x_value, y_value),
            xytext=(7, 5),
            textcoords="offset points",
            fontweight="bold",
        )
        if method != "base" and base_y is not None:
            axis.annotate(
                "",
                xy=(x_value, y_value),
                xytext=(base_x, base_y),
                arrowprops={"arrowstyle": "->", "color": COLORS[method], "lw": 2},
            )
    axis.set_xlabel(f"{task} {metric} (%)", fontweight="bold")
    axis.set_ylabel("Prior Tasks Average (%)", fontweight="bold")
    axis.set_title("Performance–Forgetting Trade-off", fontweight="bold")
    figure.savefig(report_dir / "performance_plot_5_models.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_prior_breakdown(report_dir, priors):
    labels = [label for _, label in PRIOR_TASKS]
    positions = np.arange(len(labels))
    width = 0.8 / len(METHODS)
    figure, axis = plt.subplots(figsize=(12, 6))
    for index, method in enumerate(METHODS):
        scores = [
            0.0 if priors[method][key] is None else priors[method][key] * 100
            for key, _ in PRIOR_TASKS
        ]
        axis.bar(
            positions - 0.4 + width / 2 + index * width,
            scores,
            width,
            label=METHOD_LABELS[method],
            color=COLORS[method],
        )
    axis.set_ylabel("Score (%)", fontweight="bold")
    axis.set_title("Prior Tasks Performance Breakdown", fontweight="bold")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.legend()
    figure.savefig(report_dir / "prior_tasks_breakdown.png", dpi=300, bbox_inches="tight")
    plt.close(figure)


def main():
    args = parse_args()
    report_dir = Path(args.report_dir)
    downstream_results = {
        method: read_json(report_dir / method / "downstream_results.json")
        for method in METHODS
    }
    prior_summaries = {
        method: read_json(report_dir / method / "previous_capabilities_summary.json")
        for method in METHODS
    }
    metric = downstream_results["base"]["primary_metric"]
    downstream = {
        method: float(downstream_results[method][metric]) for method in METHODS
    }
    priors = {
        method: {
            key: numeric_value(prior_summaries[method], key) for key, _ in PRIOR_TASKS
        }
        for method in METHODS
    }
    averages = {method: mean_available(priors[method].values()) for method in METHODS}

    def delta(method, values):
        if values[method] is None or values["base"] is None:
            return None
        return values[method] - values["base"]

    mmlu = {method: priors[method]["mmlu"] for method in METHODS}
    summary = {
        "task": args.task,
        "downstream_metric": metric,
        "downstream": downstream,
        "downstream_delta_vs_base": {
            method: downstream[method] - downstream["base"] for method in ("full", "lora")
        },
        "prior_tasks": priors,
        "prior_average": averages,
        "prior_average_delta_vs_base": {
            method: delta(method, averages) for method in ("full", "lora")
        },
        "mmlu_delta_vs_base": {
            method: delta(method, mmlu) for method in ("full", "lora")
        },
    }
    summary["criteria"] = {
        "full_improves_downstream": summary["downstream_delta_vs_base"]["full"] > 0,
        "lora_improves_downstream": summary["downstream_delta_vs_base"]["lora"] > 0,
        "full_reduces_mmlu": (
            summary["mmlu_delta_vs_base"]["full"] is not None
            and summary["mmlu_delta_vs_base"]["full"] < 0
        ),
        "lora_reduces_mmlu": (
            summary["mmlu_delta_vs_base"]["lora"] is not None
            and summary["mmlu_delta_vs_base"]["lora"] < 0
        ),
    }
    (report_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_markdown(report_dir, args.task, metric, downstream, priors, averages, summary)
    save_image_table(report_dir, args.task, metric, downstream, priors, averages)
    save_tradeoff_plot(report_dir, args.task, metric, downstream, averages)
    save_prior_breakdown(report_dir, priors)

    stale_plot = report_dir / "comparison.png"
    if stale_plot.exists():
        stale_plot.unlink()
    print(f"Generated report in {report_dir}")


if __name__ == "__main__":
    main()
