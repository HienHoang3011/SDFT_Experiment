import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def load_json(path):
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file {path}: {e}")
        return None

def extract_metrics(new_task_path, prior_task_path):
    new_res = load_json(new_task_path)
    prior_res = load_json(prior_task_path)

    # SQuAD metric from eval_cf_task.py is usually "f1" or "exact_match"
    new_acc = 0.0
    if new_res:
        new_acc = new_res.get("f1", 0.0) * 100

    prior_acc = 0.0
    extracted = {}
    if prior_res:
        # Lấy Accuracy từ closed_generation cho MMLU, PIQA, BoolQ
        closed_gen = prior_res.get("closed_generation", {})
        extracted["mmlu"] = closed_gen.get("mmlu", {}).get("accuracy", 0.0)
        extracted["piqa"] = closed_gen.get("piqa", {}).get("accuracy", 0.0)
        extracted["boolq"] = closed_gen.get("boolq", {}).get("accuracy", 0.0)

        # Lấy từ pure_generation cho IFEval, MATH-500, HumanEval
        pure_gen = prior_res.get("pure_generation", {})
        extracted["ifeval"] = pure_gen.get("ifeval", {}).get("prompt_level_strict_acc", 0.0)
        extracted["math500"] = pure_gen.get("math500", {}).get("flexible_match", 0.0)
        extracted["humaneval"] = pure_gen.get("humaneval", {}).get("pass@1", 0.0)

        prior_acc = prior_res.get("overall_prior_avg", 0.0)

    return new_acc, prior_acc, extracted

def print_table(models):
    print("\n" + "=" * 80)
    print(f"{'MÔ HÌNH':<20} | {'SQUAD F1 (%)':<25} | {'PRIOR TASKS AVG (%)':<25}")
    print("-" * 80)
    for name, data in models.items():
        new_acc = data["New Task Accuracy"]
        prior_acc = data["Prior Tasks Performance"] * 100
        print(f"{name:<20} | {new_acc:<25.2f} | {prior_acc:<25.2f}")
    print("=" * 80 + "\n")

def save_image_table(models, output_dir):
    categories = ["mmlu", "piqa", "boolq", "ifeval", "math500", "humaneval"]
    labels = ["MMLU", "PIQA", "BoolQ", "IFEval", "MATH-500", "HumanEval"]

    # Chuẩn bị dữ liệu
    col_labels = ["Method"] + labels + ["Average Prior", "SQuAD F1"]
    cell_text = []

    for name, data in models.items():
        row = [name]
        details = data.get("Prior Details") or {}
        for cat in categories:
            val = details.get(cat, 0.0) * 100
            row.append(f"{val:.2f}")

        avg_prior = data.get("Prior Tasks Performance", 0.0) * 100
        row.append(f"{avg_prior:.2f}")

        new_acc = data.get("New Task Accuracy", 0.0)
        row.append(f"{new_acc:.2f}")

        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(16, 1 + 0.6 * len(models)))
    ax.axis('tight')
    ax.axis('off')

    table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)

    # Định dạng header
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
        else:
            if j == 0 or j == len(col_labels) - 1 or j == len(col_labels) - 2:
                cell.set_text_props(weight='bold')
            if i % 2 == 0:
                cell.set_facecolor('#f3f4f6')

    fig.patch.set_facecolor('white')
    path = os.path.join(output_dir, "results_table.png")
    plt.savefig(path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Đã lưu bảng kết quả hình ảnh tại: {path}")

def generate_report(base_new_task=None, base_prior_task=None,
                   sft_new_task=None, sft_prior_task=None,
                   sft_lora_new_task=None, sft_lora_prior_task=None,
                   fapm_new_task=None, fapm_prior_task=None,
                   steer_new_task=None, steer_prior_task=None,
                   steer_full_new_task=None, steer_full_prior_task=None,
                   output_dir="reports"):

    os.makedirs(output_dir, exist_ok=True)
    models = {}

    b_new, b_prior, b_prior_res = extract_metrics(base_new_task, base_prior_task)
    if b_new > 0.0 or b_prior > 0.0:
        models["Base Model"] = {"New Task Accuracy": b_new, "Prior Tasks Performance": b_prior, "Prior Details": b_prior_res}

    sft_new, sft_prior, sft_prior_res = extract_metrics(sft_new_task, sft_prior_task)
    if sft_new > 0.0 or sft_prior > 0.0:
        models["Full SFT"] = {"New Task Accuracy": sft_new, "Prior Tasks Performance": sft_prior, "Prior Details": sft_prior_res}

    sft_lora_new, sft_lora_prior, sft_lora_prior_res = extract_metrics(sft_lora_new_task, sft_lora_prior_task)
    if sft_lora_new > 0.0 or sft_lora_prior > 0.0:
        models["SFT LoRA"] = {"New Task Accuracy": sft_lora_new, "Prior Tasks Performance": sft_lora_prior, "Prior Details": sft_lora_prior_res}

    fapm_new, fapm_prior, fapm_prior_res = extract_metrics(fapm_new_task, fapm_prior_task)
    if fapm_new > 0.0 or fapm_prior > 0.0:
        models["FAPM"] = {"New Task Accuracy": fapm_new, "Prior Tasks Performance": fapm_prior, "Prior Details": fapm_prior_res}

    steer_new, steer_prior, steer_prior_res = extract_metrics(steer_new_task, steer_prior_task)
    if steer_new > 0.0 or steer_prior > 0.0:
        models["Steered SFT"] = {"New Task Accuracy": steer_new, "Prior Tasks Performance": steer_prior, "Prior Details": steer_prior_res}

    steer_full_new, steer_full_prior, steer_full_prior_res = extract_metrics(steer_full_new_task, steer_full_prior_task)
    if steer_full_new > 0.0 or steer_full_prior > 0.0:
        models["Steered SFT Full"] = {"New Task Accuracy": steer_full_new, "Prior Tasks Performance": steer_full_prior, "Prior Details": steer_full_prior_res}

    if not models:
        print("CẢNH BÁO: Không có dữ liệu của bất kỳ model nào để vẽ biểu đồ.")
        return

    print_table(models)

    try:
        save_image_table(models, output_dir)
    except Exception as e:
        print(f"Lỗi khi vẽ bảng ảnh: {e}")

    try:
        colors = {"Base Model": "#999999", "Full SFT": "#d62728", "SFT LoRA": "#ff7f0e", "FAPM": "#1f77b4", "Steered SFT": "#2ca02c", "Steered SFT Full": "#9467bd"}

        # Plot 1: Scatter plot (New Task vs Prior Tasks)
        plt.figure(figsize=(9, 6))
        plt.style.use('bmh')
        ax = plt.gca()
        ax.set_facecolor('#f4f4f4')

        base_x = models.get("Base Model", {}).get("New Task Accuracy", None)
        base_y = models.get("Base Model", {}).get("Prior Tasks Performance", None)
        if base_y is not None:
            base_y *= 100

        for name, data in models.items():
            x = data["New Task Accuracy"]
            y = data["Prior Tasks Performance"] * 100

            plt.scatter(x, y, color=colors.get(name, "blue"), s=80, zorder=5, label=name)

            offset_x = 0.5
            offset_y = 0.2
            if name == "Base Model":
                plt.text(x - offset_x, y + offset_y, name, fontsize=10, ha='right', va='bottom', fontweight='bold')
            else:
                plt.text(x + offset_x, y, name, fontsize=10, ha='left', va='center', fontweight='bold')

            if name != "Base Model" and base_x is not None and base_y is not None:
                if (x != base_x or y != base_y):
                    ax.annotate('', xy=(x, y), xytext=(base_x, base_y),
                                arrowprops=dict(facecolor=colors.get(name, "blue"), edgecolor=colors.get(name, "blue"),
                                                width=2.5, headwidth=9, headlength=12, alpha=0.7, shrink=0.03),
                                zorder=3)

        plt.xlabel("SQuAD F1 (%)", fontsize=12, fontweight='bold')
        plt.ylabel("Prior Tasks Performance (%)", fontsize=12, fontweight='bold')
        plt.title("Performance Comparison (SQuAD vs Prior)", fontsize=14, fontweight='bold')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        all_x = [m["New Task Accuracy"] for m in models.values()]
        all_y = [m["Prior Tasks Performance"] * 100 for m in models.values()]
        if all_x and all_y:
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)

            plt.xlim(min(all_x) - max(2, x_range*0.2), max(all_x) + max(5, x_range*0.3))
            plt.ylim(min(all_y) - max(2, y_range*0.2), max(all_y) + max(2, y_range*0.2))

        plot_path = os.path.join(output_dir, "performance_plot_5_models.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Đã lưu biểu đồ tổng quát thành công tại: {plot_path}")

        # Plot 2: Bar chart for prior tasks breakdown
        models_with_details = {k: v for k, v in models.items() if v["Prior Details"] is not None}
        if len(models_with_details) > 0:
            categories = ["mmlu", "piqa", "boolq", "ifeval", "math500", "humaneval"]
            labels = ["MMLU", "PIQA", "BoolQ", "IFEval", "MATH-500", "HumanEval"]

            x_pos = np.arange(len(labels))
            num_models = len(models_with_details)
            total_width = 0.8
            width = total_width / max(1, num_models)

            plt.figure(figsize=(14, 6))
            plt.style.use('bmh')
            ax = plt.gca()
            ax.set_facecolor('#f4f4f4')

            start_x = x_pos - (total_width / 2) + (width / 2)

            rects_list = []
            for idx, (name, data) in enumerate(models_with_details.items()):
                details = data["Prior Details"]
                scores = [details.get(c, 0.0) * 100 for c in categories]
                rects = ax.bar(start_x + idx * width, scores, width, label=name, color=colors.get(name, "blue"))
                rects_list.append(rects)

            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title('Prior Tasks Performance Breakdown', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=11)
            ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    if height > 0:
                        ax.annotate(f'{height:.1f}',
                                    xy=(rect.get_x() + rect.get_width() / 2, height),
                                    xytext=(0, 3),
                                    textcoords="offset points",
                                    ha='center', va='bottom', fontsize=8, rotation=90)

            for rects in rects_list:
                autolabel(rects)

            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            breakdown_path = os.path.join(output_dir, "prior_tasks_breakdown.png")
            plt.savefig(breakdown_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Đã lưu biểu đồ chi tiết các task cũ tại: {breakdown_path}")

    except ImportError:
        print("Không tìm thấy thư viện matplotlib hoặc numpy để vẽ biểu đồ. Vui lòng chạy: pip install matplotlib numpy")
    except Exception as e:
        print(f"Có lỗi xảy ra khi vẽ biểu đồ: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate prior-capabilities report")
    parser.add_argument("--base_new_task", required=False)
    parser.add_argument("--base_prior_task", required=False)
    parser.add_argument("--sft_new_task", required=False)
    parser.add_argument("--sft_prior_task", required=False)
    parser.add_argument("--sft_lora_new_task", required=False)
    parser.add_argument("--sft_lora_prior_task", required=False)
    parser.add_argument("--fapm_new_task", required=False)
    parser.add_argument("--fapm_prior_task", required=False)
    parser.add_argument("--steer_new_task", required=False)
    parser.add_argument("--steer_prior_task", required=False)
    parser.add_argument("--steer_full_new_task", required=False)
    parser.add_argument("--steer_full_prior_task", required=False)
    parser.add_argument("--output_dir", default="reports")
    args = parser.parse_args()

    if args.base_prior_task or args.sft_prior_task:
        generate_report(
            base_new_task=args.base_new_task,
            base_prior_task=args.base_prior_task,
            sft_new_task=args.sft_new_task,
            sft_prior_task=args.sft_prior_task,
            sft_lora_new_task=args.sft_lora_new_task,
            sft_lora_prior_task=args.sft_lora_prior_task,
            fapm_new_task=args.fapm_new_task,
            fapm_prior_task=args.fapm_prior_task,
            steer_new_task=args.steer_new_task,
            steer_prior_task=args.steer_prior_task,
            steer_full_new_task=args.steer_full_new_task,
            steer_full_prior_task=args.steer_full_prior_task,
            output_dir=args.output_dir
        )
    else:
        pass
