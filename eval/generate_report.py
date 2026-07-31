import json
import os
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
    
    new_acc = new_res.get("accuracy", 0.0) * 100 if new_res else 0.0
    prior_acc = prior_res.get("average_score", 0.0) * 100 if prior_res else 0.0
    
    return new_acc, prior_acc, prior_res

def print_table(models):
    print("\n" + "=" * 75)
    print(f"{'MÔ HÌNH':<15} | {'NEW TASK ACCURACY (%)':<25} | {'PRIOR TASKS PERFORMANCE (%)':<25}")
    print("-" * 75)
    for name, data in models.items():
        new_acc = data["New Task Accuracy"]
        prior_acc = data["Prior Tasks Performance"]
        print(f"{name:<15} | {new_acc:<25.2f} | {prior_acc:<25.2f}")
    print("=" * 75 + "\n")

def generate_report(base_new_task=None, base_prior_task=None, 
                   sdft_new_task=None, sdft_prior_task=None, 
                   sft_new_task=None, sft_prior_task=None,
                   steer_new_task=None, steer_prior_task=None,
                   output_dir="reports"):
    
    os.makedirs(output_dir, exist_ok=True)
    models = {}
    
    b_new, b_prior, b_prior_res = extract_metrics(base_new_task, base_prior_task)
    if b_new > 0.0 or b_prior > 0.0:
        models["Base Model"] = {"New Task Accuracy": b_new, "Prior Tasks Performance": b_prior, "Prior Details": b_prior_res}
    
    s_new, s_prior, s_prior_res = extract_metrics(sdft_new_task, sdft_prior_task)
    if s_new > 0.0 or s_prior > 0.0:
        models["SDFT"] = {"New Task Accuracy": s_new, "Prior Tasks Performance": s_prior, "Prior Details": s_prior_res}

    sft_new, sft_prior, sft_prior_res = extract_metrics(sft_new_task, sft_prior_task)
    if sft_new > 0.0 or sft_prior > 0.0:
        models["SFT"] = {"New Task Accuracy": sft_new, "Prior Tasks Performance": sft_prior, "Prior Details": sft_prior_res}
        
    steer_new, steer_prior, steer_prior_res = extract_metrics(steer_new_task, steer_prior_task)
    if steer_new > 0.0 or steer_prior > 0.0:
        models["Steered SFT"] = {"New Task Accuracy": steer_new, "Prior Tasks Performance": steer_prior, "Prior Details": steer_prior_res}
        
    if not models:
        print("CẢNH BÁO: Không có dữ liệu của bất kỳ model nào để vẽ biểu đồ.")
        return

    print_table(models)
    
    try:
        colors = {"Base Model": "#999999", "SDFT": "#2b7bba", "SFT": "#d62728", "Steered SFT": "#2ca02c"}
        
        # Plot 1: Scatter plot
        plt.figure(figsize=(9, 6))
        plt.style.use('bmh')
        ax = plt.gca()
        ax.set_facecolor('#f4f4f4')
        
        base_x = models.get("Base Model", {}).get("New Task Accuracy", None)
        base_y = models.get("Base Model", {}).get("Prior Tasks Performance", None)
        
        for name, data in models.items():
            x = data["New Task Accuracy"]
            y = data["Prior Tasks Performance"]
            
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
                
        plt.xlabel("New Task Accuracy (%)", fontsize=12, fontweight='bold')
        plt.ylabel("Prior Tasks Performance (%)", fontsize=12, fontweight='bold')
        plt.title("Performance Comparison", fontsize=14, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        all_x = [m["New Task Accuracy"] for m in models.values()]
        all_y = [m["Prior Tasks Performance"] for m in models.values()]
        if all_x and all_y:
            x_range = max(all_x) - min(all_x)
            y_range = max(all_y) - min(all_y)
            
            plt.xlim(min(all_x) - max(2, x_range*0.2), max(all_x) + max(5, x_range*0.3))
            plt.ylim(min(all_y) - max(2, y_range*0.2), max(all_y) + max(2, y_range*0.2))
        
        plot_path = os.path.join(output_dir, "sdft_performance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Đã lưu biểu đồ tổng quát thành công tại: {plot_path}")

        # Plot 2: Bar chart
        models_with_details = {k: v for k, v in models.items() if v["Prior Details"] is not None}
        if len(models_with_details) > 0:
            categories = ["hellaswag", "mmlu", "truthfulqa_mc2", "winogrande", "ifeval", "humaneval", "truthfulqa_gen"]
            labels = ["HellaSwag", "MMLU", "TruthfulQA", "WinoGrande", "IFEval", "HumanEval", "TruthfulQA Gen"]
            
            x_pos = np.arange(len(labels))
            num_models = len(models_with_details)
            total_width = 0.8
            width = total_width / num_models
            
            plt.figure(figsize=(12, 6))
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
            ax.legend()
            
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
    generate_report(
        base_new_task="outputs/base_science_eval/eval_results.json",
        base_prior_task="outputs/base_prior_eval/previous_capabilities_summary.json",
        sdft_new_task="outputs/sdft_science_eval/eval_results.json",
        sdft_prior_task="outputs/sdft_prior_eval/previous_capabilities_summary.json",
        sft_new_task="outputs/sft_science_eval/eval_results.json",
        sft_prior_task="outputs/sft_prior_eval/previous_capabilities_summary.json",
        steer_new_task="outputs/steer_science_eval/eval_results.json",
        steer_prior_task="outputs/steer_prior_eval/previous_capabilities_summary.json",
    )
