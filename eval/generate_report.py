import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_json(path):
    """Đọc file json, trả về dict hoặc None nếu không tồn tại."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Lỗi khi đọc file {path}: {e}")
        return None

def extract_metrics(new_task_path, prior_task_path):
    """Trích xuất accuracy từ 2 file kết quả."""
    new_res = load_json(new_task_path)
    prior_res = load_json(prior_task_path)
    
    # Lấy điểm, giả sử điểm trong khoảng [0, 1], ta nhân 100 để đổi ra %
    new_acc = new_res.get("accuracy", 0.0) * 100 if new_res else 0.0
    prior_acc = prior_res.get("average_score", 0.0) * 100 if prior_res else 0.0
    
    return new_acc, prior_acc, prior_res

def print_table(models):
    """In bảng so sánh ra terminal."""
    print("\n" + "=" * 75)
    print(f"{'MÔ HÌNH':<15} | {'NEW TASK ACCURACY (%)':<25} | {'PRIOR TASKS PERFORMANCE (%)':<25}")
    print("-" * 75)
    for name, data in models.items():
        new_acc = data["New Task Accuracy"]
        prior_acc = data["Prior Tasks Performance"]
        print(f"{name:<15} | {new_acc:<25.2f} | {prior_acc:<25.2f}")
    print("=" * 75 + "\n")

def generate_report(base_new_task, base_prior_task, 
                   sdft_new_task, sdft_prior_task, 
                   output_dir="reports"):
    """
    Hàm chính tạo báo cáo so sánh.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    models = {}
    
    # 1. Load Base Model
    b_new, b_prior, b_prior_res = extract_metrics(base_new_task, base_prior_task)
    if b_new == 0.0 and b_prior == 0.0:
        print("CẢNH BÁO: Không tìm thấy dữ liệu Base Model, sử dụng điểm giả định (0.0).")
    models["Base Model"] = {"New Task Accuracy": b_new, "Prior Tasks Performance": b_prior, "Prior Details": b_prior_res}
    
    # 2. Load SDFT Model
    s_new, s_prior, s_prior_res = extract_metrics(sdft_new_task, sdft_prior_task)
    models["SDFT"] = {"New Task Accuracy": s_new, "Prior Tasks Performance": s_prior, "Prior Details": s_prior_res}
        
    # In dạng bảng
    print_table(models)
    
    # Vẽ dạng biểu đồ
    try:
        colors = {"Base Model": "#999999", "SDFT": "#2b7bba"}
        
        # Plot 1: Scatter plot (Overall)
        plt.figure(figsize=(9, 6))
        plt.style.use('bmh')
        ax = plt.gca()
        ax.set_facecolor('#f4f4f4')
        
        base_x = models["Base Model"]["New Task Accuracy"]
        base_y = models["Base Model"]["Prior Tasks Performance"]
        
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
            
            if name != "Base Model" and (x != base_x or y != base_y):
                ax.annotate('', xy=(x, y), xytext=(base_x, base_y),
                            arrowprops=dict(facecolor=colors.get(name, "blue"), edgecolor=colors.get(name, "blue"), 
                                            width=2.5, headwidth=9, headlength=12, alpha=0.7, shrink=0.03),
                            zorder=3)
                
        plt.xlabel("New Task Accuracy (%)", fontsize=12, fontweight='bold')
        plt.ylabel("Prior Tasks Performance (%)", fontsize=12, fontweight='bold')
        plt.title("Performance Comparison (SDFT vs Base)", fontsize=14, fontweight='bold')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        all_x = [m["New Task Accuracy"] for m in models.values()]
        all_y = [m["Prior Tasks Performance"] for m in models.values()]
        x_range = max(all_x) - min(all_x)
        y_range = max(all_y) - min(all_y)
        
        plt.xlim(min(all_x) - max(2, x_range*0.2), max(all_x) + max(5, x_range*0.3))
        plt.ylim(min(all_y) - max(2, y_range*0.2), max(all_y) + max(2, y_range*0.2))
        
        plot_path = os.path.join(output_dir, "sdft_performance_plot.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Đã lưu biểu đồ tổng quát thành công tại: {plot_path}")

        # Plot 2: Bar chart (Prior Tasks Breakdown)
        if b_prior_res and s_prior_res:
            categories = ["hellaswag", "mmlu", "truthfulqa_mc2", "winogrande", "ifeval", "humaneval", "truthfulqa_gen"]
            labels = ["HellaSwag", "MMLU", "TruthfulQA", "WinoGrande", "IFEval", "HumanEval", "TruthfulQA Gen"]
            
            b_scores = [b_prior_res.get(c, 0.0) * 100 for c in categories]
            s_scores = [s_prior_res.get(c, 0.0) * 100 for c in categories]
            
            x_pos = np.arange(len(labels))
            width = 0.35
            
            plt.figure(figsize=(10, 6))
            plt.style.use('bmh')
            ax = plt.gca()
            ax.set_facecolor('#f4f4f4')
            
            rects1 = ax.bar(x_pos - width/2, b_scores, width, label='Base Model', color=colors["Base Model"])
            rects2 = ax.bar(x_pos + width/2, s_scores, width, label='SDFT', color=colors["SDFT"])
            
            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title('Prior Tasks Performance Breakdown', fontsize=14, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(labels, fontsize=11)
            ax.legend()
            
            def autolabel(rects):
                for rect in rects:
                    height = rect.get_height()
                    ax.annotate(f'{height:.1f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
            
            autolabel(rects1)
            autolabel(rects2)
            
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
    )
