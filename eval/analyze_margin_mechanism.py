"""Combine objective math, training traces, and held-out checkpoint diagnostics."""
import argparse
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=Path('artifacts/samsum/fair-methods-2epoch-log25-tmux'))
    args = parser.parse_args()
    root = args.artifact_root
    output = root / 'reports/margin-mechanism'
    output.mkdir(parents=True, exist_ok=True)
    vocab_size = read(Path('artifacts/base/Qwen2-7B/config.json'))['vocab_size']
    margin = math.log(2.5)
    # Two distributions both satisfy the exact same logit-margin criterion.
    sharp = np.array([0.99, 0.009, 0.001])
    diffuse = np.array([2.5, 1., 1.]) / 4.5
    def entropy(p):
        return float(-(p*np.log(p)).sum())
    counterexample = {'margin': margin, 'sharp': sharp.tolist(), 'diffuse': diffuse.tolist(), 'sharp_entropy': entropy(sharp), 'diffuse_entropy': entropy(diffuse), 'kl_sharp_to_diffuse': float((sharp*np.log(sharp/diffuse)).sum()), 'kl_diffuse_to_sharp': float((diffuse*np.log(diffuse/sharp)).sum()), 'vocabulary_size': vocab_size, 'minimum_gt_probability_at_margin': 2.5/(2.5+vocab_size-1), 'note': 'Both distributions have zero self-steering loss (within floating-point precision); neither anchors a base distribution.'}
    (output/'objective_counterexample.json').write_text(json.dumps(counterexample,indent=2)+'\n')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x=np.arange(3)
    axes[0].bar(x-0.18, sharp, width=0.36, label='Sharp, margin satisfied')
    axes[0].bar(x+0.18, diffuse, width=0.36, label='Diffuse, margin satisfied')
    axes[0].set_xticks(x, ['Reference', 'Other A', 'Other B'])
    axes[0].set_ylabel('Probability')
    axes[0].set_title('Same zero steering loss, different distribution')
    axes[0].legend(fontsize=8)
    sizes=np.geomspace(2,vocab_size,150)
    axes[1].loglog(sizes, 2.5/(2.5+sizes-1))
    axes[1].set_xlabel('Vocabulary size')
    axes[1].set_ylabel('Minimum reference-token probability')
    axes[1].set_title('Max-logit margin does not control total tail mass')
    axes[1].grid(alpha=0.3)
    fig.savefig(output/'objective_mechanism.png',dpi=160)
    plt.close(fig)
    traces={}
    fig,axes=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for name in ('steer_lora','steer_full'):
        history=read(root/'models'/name/'trainer_state.json')['log_history']
        traces[name]={}
        for axis,key,title in zip(axes,('train_steer_target_kl','train_steer_activated_fraction','eval_probe_kl_base_to_ft'),('Self-steering KL during training','Fraction requiring margin boost','Held-out KL(base ∥ checkpoint)')):
            points=[(row['step'],row[key]) for row in history if key in row]
            traces[name][key]={'first':points[:3],'last':points[-3:]}
            axis.plot([p[0] for p in points],[p[1] for p in points],label=name)
            axis.set_title(title)
            axis.set_xlabel('Optimizer step')
            axis.grid(alpha=0.25)
            axis.legend()
    fig.savefig(output/'training_mechanism.png',dpi=160)
    plt.close(fig)
    (output/'training_trace_summary.json').write_text(json.dumps(traces,indent=2)+'\n')
    lines=['# Phân tích cơ chế quên của Steer margin loss','', '## Kết luận có thể kiểm chứng','', 'Loss hiện tại không dùng base làm reference. Nó so distribution hiện tại với chính distribution đó sau khi nâng logit ground-truth lên max(logit đối thủ) + log(2.5). Vì target được detach, gradient tương đương cross-entropy nhân hệ số phụ thuộc token; đạt margin thì hệ số bằng 0. Đây là cơ chế có thể chứng minh từ code, nhưng chưa tự nó chứng minh nguyên nhân duy nhất của mọi giảm điểm benchmark.','', '## Công thức gradient','', 'Gọi p = softmax(z), y là token đúng, δ = max(0, max(z khác y) + margin − z_y), và Z = 1 + (exp(δ) − 1)p_y. Target q có q_y = exp(δ)p_y/Z và q_j = p_j/Z với j khác y. Với q được detach:','', '`∂ KL(q || p) / ∂z = p − q = (1 − 1/Z) × (p − one_hot(y))`','', 'Vì vậy Steer tái trọng số gradient CE theo token, không có gradient giữ base. Một token đã đạt margin có loss/gradient bằng 0 dù distribution của nó khác base. Khi các token khác tiếp tục cập nhật tham số dùng chung, distribution của token này vẫn có thể đổi.','', '## Margin và xác suất toàn vocabulary','', f'Điều kiện margin chỉ ràng buộc đối thủ lớn nhất. Với vocabulary {vocab_size:,} token, đáp án vẫn có thể chỉ có xác suất {counterexample["minimum_gt_probability_at_margin"]:.8f} khi mọi đối thủ có logit bằng nhau và margin vừa đủ. Điều này là phản ví dụ toán học, không phải khẳng định model thực tế đang ở cực trị này.','', f'Hai distribution trong `objective_counterexample.json` đều đạt margin và có zero self-steering loss; entropy lần lượt {counterexample["sharp_entropy"]:.4f} và {counterexample["diffuse_entropy"]:.4f}; KL giữa chúng vẫn khác 0.','', '## Không thể kết luận loss đã tắt trên mọi token','', 'Log cuối của train cho thấy khoảng 46–47% token vẫn cần margin boost, và self-steering KL còn khoảng 0.13–0.14. Vì vậy giả thuyết “loss toàn bộ về 0 rồi model trôi” không phù hợp với log. Dropout LoRA 0.3 còn được bật lúc train; activation fraction lúc train không tương đương evaluation không dropout.','']
    benchmark=read(root/'stable-five/summary.json')
    lines += ['## Quên năng lực đo bằng benchmark có sẵn','','| Task | Base | Steer LoRA | Steer Full |','|---|---:|---:|---:|']
    for task in benchmark['tasks']:
        scores={row['model']:row['score'] for row in benchmark['rows'] if row['task']==task}
        if all(name in scores for name in ('base','steer_lora','steer_full')):
            lines.append(f"| {task} | {100*scores['base']:.2f} | {100*scores['steer_lora']:.2f} | {100*scores['steer_full']:.2f} |")
    for corpus in ('samsum','boolq','boolq-chat'):
        path=output/corpus/'summary.json'
        if not path.exists():
            lines += ['',f'## {corpus}: diagnostic run pending','']
            continue
        report=read(path)
        lines += ['',f'## Distribution trên {corpus}','','| Model | KL(base ∥ model) | KL(model ∥ base) | Entropy | Off-target KL(base ∥ model) | Margin active % |','|---|---:|---:|---:|---:|---:|']
        for row in report['rows']:
            lines.append(f"| {row['model']} | {row['kl_base_to_model_nats_per_token']:.4f} | {row['kl_model_to_base_nats_per_token']:.4f} | {row['model_entropy_nats']:.4f} | {row['off_target_kl_base_to_model']:.4f} | {100*row['model_margin_active']:.2f} |")
        if corpus.startswith('boolq'):
            lines += ['', 'BoolQ raw là đối chứng định dạng; BoolQ chat dùng ChatML như benchmark đã lưu. Cả hai dùng single-token no/yes. Margin ở đây là phép đo giả định trên đáp án BoolQ, không phải loss đã train trên BoolQ.','','| Model | Accuracy % | Predict yes % | ROC AUC | Gold no: yes-logit shift | Gold yes: yes-logit shift |','|---|---:|---:|---:|---:|---:|']
            for row in report['rows']:
                def shift(group):
                    means=row['groups'][group]['means']
                    return means['model_yes_minus_no_logit']-means['base_yes_minus_no_logit']
                lines.append(f"| {row['model']} | {100*row['accuracy']:.3f} | {100*row['predicted_yes_fraction']:.3f} | {row.get('yes_no_roc_auc',float('nan')):.4f} | {shift('gold_no'):.4f} | {shift('gold_yes'):.4f} |")
    if (output/'boolq/summary.json').exists() and (output/'boolq-chat/summary.json').exists():
        raw = {row['model']:row for row in read(output/'boolq/summary.json')['rows']}
        chat = {row['model']:row for row in read(output/'boolq-chat/summary.json')['rows']}
        names=list(raw)
        fig, axes=plt.subplots(1,2,figsize=(13,4),constrained_layout=True)
        positions=np.arange(len(names))
        for axis, key, title in zip(axes,('accuracy','predicted_yes_fraction'),('BoolQ accuracy','Fraction choosing yes')):
            axis.bar(positions-0.18,[100*raw[name][key] for name in names],width=0.36,label='Raw')
            axis.bar(positions+0.18,[100*chat[name][key] for name in names],width=0.36,label='ChatML')
            axis.set_xticks(positions,names,rotation=25,ha='right')
            axis.set_ylim(0,105)
            axis.set_ylabel('%')
            axis.set_title(title)
            axis.grid(axis='y',alpha=0.25)
            axis.legend()
        fig.savefig(output/'boolq_prompt_interaction.png',dpi=160)
        plt.close(fig)
        lines += ['', '## Phát hiện chính trên BoolQ', '', 'Cùng checkpoint, cùng 3270 câu hỏi và nhãn, thay định dạng prompt từ raw sang ChatML làm quyết định yes/no của Steer gần như chỉ còn yes. Đây là can thiệp định dạng đầu vào, không phải một causal ablation của quá trình training.']
        for name in ('steer_lora','steer_full'):
            raw_delta=100*(raw[name]['accuracy']-raw['base']['accuracy'])
            chat_delta=100*(chat[name]['accuracy']-chat['base']['accuracy'])
            lines.append(f"- {name}: raw {100*raw[name]['accuracy']:.3f}% (so base {raw_delta:+.3f} điểm %); ChatML {100*chat[name]['accuracy']:.3f}% (so base {chat_delta:+.3f} điểm %); chọn yes {100*chat[name]['predicted_yes_fraction']:.3f}%.")
        lines += ['', 'Accuracy gần tỷ lệ nhãn yes là quyết định majority-class sau khi so likelihood no/yes; không có nghĩa rằng greedy generation luôn sinh chữ yes. ROC AUC vẫn trên 0.8 trên ChatML, cho thấy một phần tín hiệu phân biệt còn tồn tại, nhưng AUC thấp hơn base nên đây không chỉ là dịch ngưỡng hoàn toàn thuần túy. Raw AUC khoảng 0.93 là bằng chứng chống lại diễn giải kiến thức BoolQ bị mất hoàn toàn.', '', 'Diagnostic dùng fp32 log-softmax; benchmark đã lưu dùng bf16 softmax và cộng continuation newline chung. Những khác biệt nhỏ/ties ở base và SFT/LoRA được giữ tách khỏi sự sụp quyết định của Steer; prediction của hai Steer tái hiện đúng từng mẫu benchmark cũ.']

    samsum_dir = output/'samsum'
    if (samsum_dir/'summary.json').exists():
        confidence_groups = {}
        lines += ['', '## Token dễ và khó trên SAMSum', '', '| Model | Nhóm theo base | Tokens | Δ P(token đúng) | Δ NLL | Δ entropy | KL(base ∥ model) |', '|---|---|---:|---:|---:|---:|---:|']
        for name in ('full_sft', 'lora', 'steer_lora', 'steer_full'):
            groups = {}
            with (samsum_dir/f'{name}.tokens.jsonl').open() as handle:
                for line in handle:
                    metrics = json.loads(line)['metrics']
                    probability = metrics['base_target_probability']
                    selected = ['base_p_lt_0.1' if probability < 0.1 else ('base_p_ge_0.9' if probability >= 0.9 else 'base_p_0.1_to_0.9')]
                    if metrics['base_margin_active'] == 0:
                        selected.append('base_already_satisfies_margin')
                    for group in selected:
                        bucket = groups.setdefault(group, {'tokens': 0, 'target_probability_delta': 0., 'target_nll_delta': 0., 'entropy_delta': 0., 'kl_base_to_model': 0.})
                        bucket['tokens'] += 1
                        bucket['target_probability_delta'] += metrics['model_target_probability'] - probability
                        bucket['target_nll_delta'] += metrics['model_target_nll_nats'] - metrics['base_target_nll_nats']
                        bucket['entropy_delta'] += metrics['model_entropy_nats'] - metrics['base_entropy_nats']
                        bucket['kl_base_to_model'] += metrics['kl_base_to_model_nats_per_token']
            confidence_groups[name] = {}
            for group, bucket in groups.items():
                means = {key: value/bucket['tokens'] for key,value in bucket.items() if key != 'tokens'}
                confidence_groups[name][group] = {'tokens': bucket['tokens'], **means}
                lines.append(f"| {name} | {group} | {bucket['tokens']} | {means['target_probability_delta']:+.4f} | {means['target_nll_delta']:+.4f} | {means['entropy_delta']:+.4f} | {means['kl_base_to_model']:.4f} |")
        (output/'confidence_groups.json').write_text(json.dumps(confidence_groups,indent=2)+'\n')
        lines += ['', 'Nhóm được cố định bằng distribution của base, cùng token/prefix giữa các model. Δ NLL âm là fit đáp án tốt hơn. Độ lệch trên nhóm base đã đạt margin chứng minh checkpoint thay đổi cả các vị trí không cần boost ở thời điểm khởi đầu; chưa xác định duy nhất nguồn gây thay đổi.']

    lines += ['', '## Cách đọc độ lệch ngoài đáp án','', 'Một thao tác margin trực tiếp chỉ đổi logit y; tỉ lệ xác suất giữa các token khác y giữ nguyên. Off-target KL tái chuẩn hóa vocabulary sau khi bỏ y. Off-target KL lớn tại checkpoint nghĩa là thay đổi không thể chỉ mô tả bằng một lần boost logit ground-truth. Nó có thể đến từ tích lũy cập nhật tham số, chuyển miền và tái trọng số token; checkpoint comparison chưa phân biệt được các nguyên nhân đó.','', '## Đối chứng để kiểm tra nguyên nhân','', 'Biến thể đã code: Steer + λ KL(base || model), dùng base cố định với adapter tắt. Chạy cùng seed, LoRA, số epoch, optimizer và dữ liệu; λ=1 là lựa chọn trước thử nghiệm, không được coi là tối ưu. Lượt control tự động đo BoolQ raw/ChatML và chất lượng SAMSum; IFEval chưa được đưa vào lượt control này. Anchor được train trên SAMSum, không có replay của BoolQ hoặc task cũ. Nếu drift giảm nhưng điểm không phục hồi, giả thuyết “quên hoàn toàn do thiếu base anchor” chưa đủ. Muốn kết luận riêng về margin, cần thêm ablation margin size và CE + anchor, cùng nhiều seed; chúng chưa được chạy.','', 'Các kết quả hiện tại là quan sát của một training seed. Bootstrap của probe không thay thế độ biến thiên giữa các training seed.']
    (output/'mechanism_report.md').write_text('\n'.join(lines)+'\n')
    print((output/'mechanism_report.md').resolve())


if __name__=='__main__':
    main()
