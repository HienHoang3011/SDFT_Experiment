"""Report the frozen-base anchor as a mechanistic control, not a tuned winner."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment_root', type=Path)
    parser.add_argument('--reference-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.experiment_root
    distribution = read(root/'reports/distribution/summary.json')
    lines = ['# Frozen-base anchor control', '', 'Objective: Steer + 1.0 × KL(frozen base || model). Same SAMSum data, 2 epochs, rank 256, alpha 512, dropout 0.3, effective batch size 64, learning rate 1e-5 and seed 42 as the existing Steer-LoRA recipe. Runtime package versions are in logs/environment.txt. This is one seed and one preselected anchor weight, not a hyperparameter sweep.', '', '| Model | KL(base ∥ model) | KL(model ∥ base) | JS | Entropy | SAMSum dev target NLL |', '|---|---:|---:|---:|---:|---:|']
    for row in distribution['rows']:
        lines.append(f"| {row['model']} | {row['kl_base_to_model_nats_per_token']:.6f} | {row['kl_model_to_base_nats_per_token']:.6f} | {row['js_nats_per_token']:.6f} | {row['model_entropy_nats']:.6f} | {row['model_target_nll_nats']:.6f} |")
    for prompt_format in ('raw','chat'):
        report=read(root/f'reports/boolq-{prompt_format}/summary.json')
        lines += ['', f'## BoolQ ({prompt_format})', '', '| Model | Accuracy % | Predicted yes % | ROC AUC | Base-correct → wrong | Base-wrong → correct |','|---|---:|---:|---:|---:|---:|']
        for row in report['rows']:
            lines.append(f"| {row['model']} | {100*row['accuracy']:.3f} | {100*row['predicted_yes_fraction']:.3f} | {row['yes_no_roc_auc']:.4f} | {row['base_correct_model_wrong']} | {row['base_wrong_model_correct']} |")
    quality=read(root/'reports/samsum/downstream_results.json')
    old=read(args.reference_root/'reports/steer_lora/downstream_results.json')
    lines += ['', '## SAMSum test quality', '', f"Existing Steer-LoRA ROUGE-L: {100*old['rougeL']:.3f}%. Anchored Steer-LoRA: {100*quality['rougeL']:.3f}%.", '', 'Interpretation: reduced KL alone is expected from the anchor and does not establish preserved capability. Inspect ChatML BoolQ recovery, raw/chat differences, and downstream quality jointly. This experiment does not isolate every effect of the margin; a causal margin claim also needs margin-size and CE-plus-anchor ablations and multiple seeds.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print((root/'REPORT.md').resolve())


if __name__ == '__main__':
    main()
