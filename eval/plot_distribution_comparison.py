"""Plot distribution metrics and example-bootstrap confidence intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report_dir', type=Path)
    args = parser.parse_args()
    report = json.loads((args.report_dir / 'summary.json').read_text())
    rows = [row for row in report['rows'] if row['model'] != 'base']
    panels = [
        ('kl_base_to_model_nats_per_token', 'KL(base ∥ model)', 'nats/token'),
        ('kl_model_to_base_nats_per_token', 'KL(model ∥ base)', 'nats/token'),
        ('js_nats_per_token', 'Jensen–Shannon', 'nats/token'),
        ('total_variation', 'Total variation', 'distance'),
        ('model_entropy_nats', 'Model entropy', 'nats/token'),
        ('model_target_nll_nats', 'Reference-answer NLL', 'nats/token'),
    ]
    colors = ['#356ea8' if not row['model'].startswith('steer') else '#c5632e' for row in rows]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, (key, title, unit) in zip(axes.flat, panels):
        means = np.array([row[key] for row in rows])
        ci = np.array([row['statistics'][key]['example_bootstrap_ci95'] for row in rows])
        axis.bar(range(len(rows)), means, color=colors, alpha=0.9)
        # Percentile bootstrap intervals need not bracket the observed estimate.
        axis.vlines(range(len(rows)), ci[:, 0], ci[:, 1], color='#252525', linewidth=1.5)
        axis.plot(range(len(rows)), ci[:, 0], '_', color='#252525')
        axis.plot(range(len(rows)), ci[:, 1], '_', color='#252525')
        axis.set_xticks(range(len(rows)), [row['model'] for row in rows], rotation=25, ha='right')
        axis.set_title(title)
        axis.set_ylabel(unit)
        axis.grid(axis='y', alpha=0.2)
        axis.set_axisbelow(True)
        if key in ('model_entropy_nats', 'model_target_nll_nats'):
            baseline = report['rows'][0][key]
            axis.axhline(baseline, color='#444444', linestyle='--', label=f'Base = {baseline:.3f}')
            axis.legend(fontsize=8)
    fig.suptitle(f"{report['dataset']}: {rows[0]['dev_examples']} examples, {rows[0]['assistant_tokens']} tokens, T={report['temperature']}\nBars: token means. Lines: 95% example-bootstrap intervals.", fontsize=13)
    for extension in ('png', 'svg'):
        path = args.report_dir / f'comparison.{extension}'
        fig.savefig(path, dpi=160)
        print(path.resolve())
    plt.close(fig)


if __name__ == '__main__':
    main()
