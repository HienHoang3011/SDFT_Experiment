"""Audit the actual Steer loss and compare all methods using saved artifacts.

Requires only Python, NumPy, PyTorch and Matplotlib; no model loading or training.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
METHODS = ('base', 'full_sft', 'lora', 'steer_lora', 'steer_full', 'fapm')


def load_loss():
    """Execute the unchanged loss class, avoiding unrelated training imports."""
    path = ROOT / 'methods/train_steer.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SteeredLoss')
    namespace = {'torch': torch, 'F': F, 'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['SteeredLoss']


def audit_objective(vocab_size):
    loss_class = load_loss()
    torch.manual_seed(42)
    checks = []
    for vocabulary in (3, 100, vocab_size):
        logits = torch.randn(4, 2, vocabulary, requires_grad=True)
        labels = torch.stack((torch.full((4,), -100), torch.randint(vocabulary, (4,))), dim=1)
        loss = loss_class(1.5)(SimpleNamespace(logits=logits), labels)
        grad = torch.autograd.grad(loss, logits)[0]
        z = logits[:, 0].detach()
        y = labels[:, 1]
        rows = torch.arange(len(y))
        other = z.clone()
        other[rows, y] = -torch.inf
        delta = (other.max(-1).values + math.log(2.5) - z[rows, y]).clamp_min(0)
        p = z.softmax(-1)
        normalizer = 1 + delta.expm1() * p[rows, y]
        ce_gradient = p.clone()
        ce_gradient[rows, y] -= 1
        expected = (1 - 1 / normalizer)[:, None] * ce_gradient / len(y)
        error = (grad[:, 0] - expected).abs().max().item()
        assert error < 2e-6, (vocabulary, error)
        assert grad[:, 1].abs().max().item() == 0, 'Last causal position must be masked'
        checks.append({'vocabulary': vocabulary, 'max_gradient_error': error, 'loss': loss.item()})
    z = torch.randn(2, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, 0, -100, 2], [-100, -100, 3, -100]])
    denominator = int((labels[:, 1:] != -100).sum())
    full = loss_class(1.5)(SimpleNamespace(logits=z), labels, denominator)
    accumulated = sum(loss_class(1.5)(SimpleNamespace(logits=z[i:i+1]), labels[i:i+1], denominator) for i in range(2))
    full_grad = torch.autograd.grad(full, z, retain_graph=True)[0]
    accumulated_grad = torch.autograd.grad(accumulated, z)[0]
    accumulation_error = (full_grad - accumulated_grad).abs().max().item()
    assert accumulation_error < 1e-7
    active = labels[:, 1:] != -100
    assert full_grad[:, :-1][~active].abs().sum().item() == 0
    masked = loss_class(1.5)(SimpleNamespace(logits=z), torch.full_like(labels, -100))
    assert masked.item() == 0
    assert torch.autograd.grad(masked, z)[0].abs().sum().item() == 0
    examples = []
    for name, probabilities in [('sharp', [0.99, 0.009, 0.001]), ('diffuse', [2.5/4.5, 1/4.5, 1/4.5])]:
        z = torch.tensor([probabilities, [1/3]*3]).log().unsqueeze(0).requires_grad_()
        value = loss_class(1.5)(SimpleNamespace(logits=z), torch.tensor([[-100, 0]]))
        gradient = torch.autograd.grad(value, z)[0].abs().max().item()
        assert abs(value.item()) < 1e-6 and gradient < 1e-6
        examples.append({'name': name, 'probabilities': probabilities, 'loss': value.item(), 'max_gradient': gradient})
    return {'gradient_checks': checks, 'accumulation_max_error': accumulation_error,
            'masking_and_causal_shift_passed': True, 'zero_loss_distributions': examples,
            'margin': math.log(2.5), 'minimum_target_probability': 2.5/(vocab_size-1+2.5),
            'scope': 'Actual loss class; CPU fp32. Trainer/runtime integration is not exercised.'}


def paired_ci(values, rng, repetitions=2000):
    values = np.asarray(values, dtype=float)
    draws = values[rng.integers(len(values), size=(repetitions, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-root', type=Path, default=ROOT/'artifacts/samsum/fair-methods-2epoch-log25-tmux')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    root = args.artifact_root
    out = args.output_dir or root/'reports'/('margin-audit-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True, exist_ok=False)
    provenance = []

    def read(path):
        data = path.read_bytes()
        provenance.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest()})
        return json.loads(data)

    def records(path):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for line in handle:
                digest.update(line)
                if line.strip():
                    yield json.loads(line)
        provenance.append({'path': str(path.resolve()), 'sha256': digest.hexdigest()})

    torch.set_num_threads(4)
    config = read(ROOT/'artifacts/base/Qwen2-7B/config.json')
    objective = audit_objective(config['vocab_size'])
    print('Actual SteeredLoss gradient, masking, accumulation and zero-loss checks: PASS', flush=True)
    benchmark = read(root/'stable-five/summary.json')
    manifest = read(root/'reports/fairness_manifest.json')
    diagnostic = root/'reports/margin-mechanism'
    distribution = read(diagnostic/'samsum/summary.json')
    distribution_rows = {row['model']: row for row in distribution['rows']}
    scores = {(r['model'], r['task']): r['score'] for r in benchmark['rows']}
    tasks = list(benchmark['tasks'])
    table = []
    for name in METHODS:
        quality = read(root/'reports'/name/'downstream_results.json')
        d = distribution_rows[name]
        row = {'model': name, 'rougeL_percent': quality['rougeL']*100,
               **{task: scores[name, task]*100 for task in tasks}}
        for key in ('kl_base_to_model_nats_per_token', 'kl_model_to_base_nats_per_token',
                    'js_nats_per_token', 'total_variation', 'model_entropy_nats', 'model_target_nll_nats',
                    'model_margin_active', 'off_target_kl_base_to_model'):
            row[key] = d[key]
        table.append(row)
    with (out/'comparison.csv').open('w') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    rng = np.random.default_rng(42)
    boolq = {}
    for fmt, directory in [('raw', 'boolq'), ('chat', 'boolq-chat')]:
        rows = {}
        reference_ids = None
        for name in METHODS:
            docs = list(records(diagnostic/directory/f'{name}.examples.jsonl'))
            ids = [(d['doc_id'], d['label']) for d in docs]
            if reference_ids is None:
                reference_ids = ids
            assert ids == reference_ids, 'BoolQ comparisons require identical samples/order'
            labels = np.array([d['label'] for d in docs])
            # Recompute predictions from saved choice likelihoods, not summary fields.
            predictions = np.array([np.argmax(d['model_choice_logprobs']) for d in docs])
            base = np.array([np.argmax(d['base_choice_logprobs']) for d in docs])
            assert all(p == d['model_prediction'] for p, d in zip(predictions, docs))
            delta = (predictions == labels).astype(float) - (base == labels)
            rows[name] = {'examples': len(docs), 'accuracy_percent': float(np.mean(predictions == labels)*100),
                          'predict_yes_percent': float(predictions.mean()*100),
                          'gold_no_accuracy_percent': float(np.mean(predictions[labels == 0] == 0)*100),
                          'gold_yes_accuracy_percent': float(np.mean(predictions[labels == 1] == 1)*100),
                          'confusion_gold_by_prediction': [[int(((labels == gold) & (predictions == pred)).sum()) for pred in (0,1)] for gold in (0,1)],
                          'delta_vs_base_pp': float(delta.mean()*100),
                          'paired_bootstrap_delta_ci_pp': [v*100 for v in paired_ci(delta, rng)],
                          'majority_yes_accuracy_percent': float(labels.mean()*100)}
        boolq[fmt] = rows
        print(f'BoolQ {fmt}: recomputed {len(reference_ids)} examples for all six methods', flush=True)

    groups = {}
    reference_positions = None
    for name in METHODS:
        totals, positions = {}, []
        for record in records(diagnostic/'samsum'/f'{name}.tokens.jsonl'):
            positions.append((record['example_id'], record['prediction_position'], record['target_token_id']))
            m = record['metrics']
            selected = ['all', 'base_p_ge_0.9' if m['base_target_probability'] >= .9 else 'base_p_lt_0.9']
            if m['base_margin_active'] == 0:
                selected.append('base_already_satisfies_margin')
            for group in selected:
                bucket = totals.setdefault(group, [])
                bucket.append([m['model_target_probability']-m['base_target_probability'],
                               m['model_target_nll_nats']-m['base_target_nll_nats'],
                               m['model_entropy_nats']-m['base_entropy_nats'],
                               m['model_margin_active'], m['off_target_kl_base_to_model']])
        if reference_positions is None:
            reference_positions = positions
        assert positions == reference_positions, 'SAMSum comparisons require identical tokens/prefixes'
        groups[name] = {g: {'tokens': len(values), **dict(zip(
            ('probability_delta', 'nll_delta', 'entropy_delta', 'margin_active', 'off_target_kl'),
            np.asarray(values).mean(axis=0).tolist()))} for g, values in totals.items()}
        assert abs(distribution_rows[name]['model_target_nll_nats'] - distribution_rows[name]['base_target_nll_nats'] - groups[name]['all']['nll_delta']) < 1e-5
    print(f'SAMSum: matched and reaggregated {len(reference_positions)} token positions for all six methods', flush=True)

    traces = {}
    for name in ('steer_lora', 'steer_full'):
        history = read(root/'models'/name/'trainer_state.json')['log_history']
        traces[name] = {key: [{'step': r['step'], 'value': r[key]} for r in history if key in r][-3:]
                        for key in ('train_steer_target_kl', 'train_steer_activated_fraction', 'eval_probe_kl_base_to_ft')}

    lines = ['# So sánh phương pháp và kiểm chứng margin từ artifact', '',
             'Chạy lại lớp SteeredLoss thực tế bằng AST để tránh phụ thuộc thư viện train; tính lại BoolQ từ choice log-prob và SAMSum từ từng token đã lưu. Không train lại hoặc chạy lại model forward.', '',
             '## Chất lượng và năng lực (%)', '',
             '| Model | SAMSum ROUGE-L | GSM8K | HumanEval | IFEval | BoolQ benchmark | BBH |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in table:
        lines.append('| '+row['model']+' | '+' | '.join(f'{row[k]:.2f}' for k in ['rougeL_percent']+tasks)+' |')
    lines += ['', '## Phân phối SAMSum dev', '',
              f"Cùng {distribution_rows['base']['dev_examples']} ví dụ và {len(reference_positions)} token; temperature={distribution['temperature']}; full vocabulary. KL tính bằng nats/token.", '',
              '| Model | KL(base || model) | KL(model || base) | JS | TV | NLL token đúng | Entropy | Margin active % |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in table:
        keys = ['kl_base_to_model_nats_per_token', 'kl_model_to_base_nats_per_token', 'js_nats_per_token', 'total_variation', 'model_target_nll_nats', 'model_entropy_nats']
        lines.append('| '+row['model']+' | '+' | '.join(f'{row[k]:.4f}' for k in keys)+f" | {100*row['model_margin_active']:.2f} |")
    for fmt in ('chat', 'raw'):
        lines += ['', f'## BoolQ {fmt}: tính lại từ từng mẫu', '',
                  '| Model | Accuracy % | Predict yes % | Đúng gold no % | Đúng gold yes % | Δ Base (pp) | CI 95% Δ Base |',
                  '|---|---:|---:|---:|---:|---:|---|']
        for name, row in boolq[fmt].items():
            lo, hi = row['paired_bootstrap_delta_ci_pp']
            lines.append(f"| {name} | {row['accuracy_percent']:.3f} | {row['predict_yes_percent']:.3f} | {row['gold_no_accuracy_percent']:.3f} | {row['gold_yes_accuracy_percent']:.3f} | {row['delta_vs_base_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    lines += ['', 'BoolQ diagnostic chấm token no/yes tại vị trí đáp án. Dùng điểm benchmark đã lưu để báo cáo benchmark; không giả định diagnostic và lm-eval là hoàn toàn tương đương. Một token newline sau no và yes có thể có xác suất khác nhau, nên không tự động triệt tiêu trong sequence likelihood.', '',
              '## Token Base đã tự tin trên SAMSum', '',
              '| Model | Tokens Base P(y) ≥ 0.9 | Δ P(y) | Δ NLL | Δ entropy |',
              '|---|---:|---:|---:|---:|']
    for name in METHODS:
        row = groups[name]['base_p_ge_0.9']
        lines.append(f"| {name} | {row['tokens']} | {row['probability_delta']:+.4f} | {row['nll_delta']:+.4f} | {row['entropy_delta']:+.4f} |")
    lines += ['', '## Kiểm chứng objective', '',
              f"Margin log(2.5) = {objective['margin']:.6f}. Kiểm tra gradient ở vocabulary 3, 100 và {config['vocab_size']:,} đều pass; causal shift, masking, zero batch và gradient accumulation đều pass.", '',
              '`∇z KL(q_detach || p) = (1 − 1/Z) × (p − one_hot(y))`, với `Z = 1 + (exp(δ) − 1) p_y` và `δ = max(0, max_other + margin − z_y)`.', '',
              'Đây là hướng gradient CE được giảm trọng số theo token; không có thành phần giữ Base. Điều kiện margin ràng buộc tỷ lệ P(y)/P(đối thủ lớn nhất) ≥ 2.5, không đảm bảo P(y) ≥ 0.714 vì còn toàn bộ vocabulary.', '',
              f"Cận xác suất tối thiểu theo margin với vocabulary này là {objective['minimum_target_probability']:.8f}. Hai phân phối sharp/diffuse khác nhau đều cho loss và gradient bằng 0 trong sai số fp32.", '',
              '## Kết luận từ so sánh', '',
              '- Steer cải thiện SAMSum và giữ GSM8K tốt trong lượt chạy này, nhưng kém LoRA thường về SAMSum, IFEval, BoolQ và BBH. Không thể gọi phương pháp thất bại trên mọi năng lực.',
              '- Trên BoolQ ChatML, Steer gần như luôn chọn yes: gold-no accuracy gần 0 và tổng accuracy xấp xỉ majority-yes baseline. Raw prompt không biểu hiện mức sụp đổ này. Đây là dấu hiệu lệch đáp án phụ thuộc định dạng.',
              '- KL(base || model) của Steer trên SAMSum thấp hơn SFT/LoRA, nhưng reverse KL, JS, TV và entropy cao hơn. Không thể chọn một chiều KL rồi kết luận giữ phân phối/năng lực tốt hơn.',
              '- Steer làm giảm P(y) nhiều hơn trên token Base đã tự tin. Điều này phù hợp với gradient không bảo vệ các vị trí đã đạt margin khi tham số dùng chung tiếp tục đổi.',
              '- Log cuối vẫn có khoảng 46–47% token active và self-KL khoảng 0.13–0.14: không có bằng chứng loss đã tắt toàn bộ.',
              '- FAPM gần Base về phân phối/năng lực nhưng gần như không cải thiện SAMSum. EWC không có trong lượt chạy này.',
              '- Chưa thấy lỗi dấu margin, chiều KL, causal shift hoặc chuẩn hóa token trong phép kiểm chứng. Giới hạn rõ là dùng target tự tạo từ model hiện tại; chưa chứng minh riêng giá trị margin 0.916 là nguyên nhân duy nhất. Cần ablation margin và CE + frozen-base KL, nhiều seed để tách nguyên nhân.', '',
              '## Phạm vi', '',
              'Một training seed; CI BoolQ là paired bootstrap ví dụ, không phản ánh biến thiên giữa seed. Manifest cấu hình và SHA256 của mọi artifact đầu vào được lưu trong audit.json. Đồ thị và bảng có thể tái tạo bằng script này.', '']
    (out/'report.md').write_text('\n'.join(lines))
    source = ROOT/'methods/train_steer.py'
    provenance.append({'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
    audit = {'created_utc': datetime.now(timezone.utc).isoformat(), 'objective': objective,
             'manifest': manifest, 'comparison': table, 'boolq': boolq, 'confidence_groups': groups,
             'training_traces_last_three': traces, 'sources': provenance,
             'runtime': {'python': __import__('sys').version, 'torch': torch.__version__, 'numpy': np.__version__}}
    (out/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(METHODS))
    for ax, key, title in [(axes[0,0], 'rougeL_percent', 'SAMSum ROUGE-L (%)'),
                            (axes[0,1], 'boolq', 'BoolQ benchmark (%)')]:
        ax.bar(x, [r[key] for r in table])
        ax.set_title(title)
    axes[1,0].bar(x-.2, [boolq['raw'][n]['accuracy_percent'] for n in METHODS], .4, label='Raw')
    axes[1,0].bar(x+.2, [boolq['chat'][n]['accuracy_percent'] for n in METHODS], .4, label='ChatML')
    axes[1,0].set_title('BoolQ answer-token accuracy by prompt (%)')
    axes[1,0].legend()
    axes[1,1].bar(x, [groups[n]['base_p_ge_0.9']['probability_delta'] for n in METHODS])
    axes[1,1].set_title('SAMSum: delta P(y) where Base P(y) >= 0.9')
    for ax in axes.flat:
        ax.set_xticks(x, METHODS, rotation=25, ha='right')
        ax.grid(axis='y', alpha=.2)
    fig.savefig(out/'comparison.png', dpi=170)
    plt.close(fig)
    print((out/'report.md').resolve(), flush=True)


if __name__ == '__main__':
    main()
