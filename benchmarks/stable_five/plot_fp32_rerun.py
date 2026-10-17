"""Render the fp32-master rerun scores in the Stable Five table style.

Base, LoRA and Steer-LoRA are unaffected by the bf16 issue and reuse the original run
(GSM8K from its 1024-token rerun). Full SFT, Steer-Full and FAPM come from the fp32 rerun,
whose GSM8K already uses 1024 tokens. A second table compares old bf16 and new fp32 scores.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "artifacts/samsum/fair-methods-2epoch-log25-tmux"
NEW = ROOT / "artifacts/samsum/fp32-master-2epoch-20261012"
TASKS = [("gsm8k_cot_zeroshot", "GSM8K · 1024 tokens"), ("humaneval_instruct", "HumanEval"),
         ("ifeval", "IFEval"), ("boolq", "BoolQ"), ("bbh_cot_zeroshot", "BBH")]
FULL = ("full_sft", "steer_full", "fapm")
LABELS = {"base": "Base", "full_sft": "Full SFT", "lora": "LoRA", "steer_lora": "Steer LoRA",
          "steer_full": "Steer Full", "fapm": "FAPM"}


def scores(path):
    if not path.exists():
        return {}
    return {(row["task"], row["model"]): row["score"] for row in json.loads(path.read_text())["rows"]}


def rouge(root, model):
    path = root / "reports" / model / "downstream_results.json"
    return json.loads(path.read_text())["rougeL"] if path.exists() else None


def new_scores():
    # Read per-model result files directly so partial runs show what has finished.
    found = {}
    for path in (NEW / "stable-five/models").glob("*/*.json"):
        payload = json.loads(path.read_text())
        found[(payload["task"], payload["model"])] = payload["score"]
    return found


def cell(value):
    return f"{value * 100:.2f}%" if value is not None else "—"


def draw(rows, columns, title, footnote, output, bold_last=False):
    fig, ax = plt.subplots(figsize=(1.9 * len(columns) + 3.5, 1.4 + 0.62 * len(rows)))
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=20, weight="bold", pad=20)
    table = ax.table(cellText=rows, colLabels=["Benchmark"] + columns, cellLoc="center", loc="center",
                     colWidths=[0.25] + [0.125] * len(columns))
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)
    for (row, column), c in table.get_celld().items():
        c.set_edgecolor("#dce4ec")
        last = bold_last and row == len(rows)
        c.set_facecolor("#17324d" if row == 0 else ("#e4eff7" if last else ("#f1f5f9" if row % 2 == 0 else "white")))
        if row == 0:
            c.get_text().set_color("white")
            c.get_text().set_weight("bold")
        elif last:
            c.get_text().set_weight("bold")
        if c.get_text().get_text() == "—":
            c.get_text().set_color("#8a99a8")
        if column == 0:
            c.get_text().set_ha("left")
    fig.text(0.045, 0.04, footnote, fontsize=10, color="#526477")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=NEW)
    args = parser.parse_args()
    old_main = scores(OLD / "stable-five/summary.json")
    old_gsm = scores(OLD / "stable-five-gsm8k-1024/summary.json")
    new = new_scores()

    def old_score(task, model):
        source = old_gsm if task == "gsm8k_cot_zeroshot" else old_main
        return source.get((task, model))

    # Base / LoRA / Steer LoRA are re-scored in the rerun environment; until a score exists
    # there, the original-run score is shown and marked with "*".
    def current(task, model):
        if (task, model) in new or model in FULL:
            return cell(new.get((task, model)))
        value = old_score(task, model)
        return cell(value) + ("*" if value is not None else "")

    def current_rouge(model):
        value = rouge(NEW, model)
        if value is not None or model in FULL:
            return cell(value)
        value = rouge(OLD, model)
        return cell(value) + ("*" if value is not None else "")

    models = ["base", "full_sft", "lora", "steer_lora", "steer_full", "fapm"]
    rows = [[label, *(current(task, m) for m in models)] for task, label in TASKS]
    rows.append(["SAMSum ROUGE-L", *(current_rouge(m) for m in models)])
    draw(rows, [LABELS[m] + ("\nfp32" if m in FULL else "") for m in models], "Stable Five · scores · fp32 rerun",
         "Scores in percent  ·  — = pending  ·  * = original-run score, re-scoring pending  ·  "
         "Full SFT / Steer Full / FAPM retrained with fp32 master weights", args.output_dir / "scores-fp32-rerun.png",
         bold_last=True)

    columns, compare = [], []
    for m in FULL:
        columns += [f"{LABELS[m]} bf16", f"{LABELS[m]} fp32"]
    for task, label in TASKS:
        compare.append([label, *(cell(v) for m in FULL for v in (old_score(task, m), new.get((task, m))))])
    compare.append(["SAMSum ROUGE-L", *(cell(v) for m in FULL for v in (rouge(OLD, m), rouge(NEW, m)))])
    draw(compare, columns, "Pure bf16 vs fp32 master weights",
         "Scores in percent  ·  — = pending  ·  bf16 = original run (weights stored in bf16); fp32 = rerun",
         args.output_dir / "scores-bf16-vs-fp32.png", bold_last=True)


if __name__ == "__main__":
    main()
