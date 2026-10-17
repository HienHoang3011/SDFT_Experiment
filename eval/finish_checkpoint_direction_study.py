"""Extend completed calibration equally across checkpoints, then summarize batches.

The wall-time target determines sample count only. It never delays GPU work
to consume a budget. Intervals observed during calibration include forward,
backward, optimizer, CPU direction metrics, and held-out probes.
"""
import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def dump(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def choose_batches(elapsed, target, round_seconds, calibration_batches, max_batches):
    if round_seconds <= 0 or not math.isfinite(round_seconds):
        raise ValueError('Invalid measured batch duration')
    # Reserve repeated loading/probe setup and 5% for runtime variation.
    setup = max(0., elapsed - calibration_batches * round_seconds)
    additional = max(0, math.floor((target - elapsed - setup) / (round_seconds * 1.05)))
    return min(max_batches, calibration_batches + additional), setup


def interval(values, rng):
    values = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    if not len(values):
        return {'n': 0, 'mean': None, 'ci95': None}
    means = values[rng.integers(0, len(values), size=(4000, len(values)))].mean(axis=1)
    return {'n': len(values), 'mean': float(values.mean()),
            'ci95': np.quantile(means, [.025, .975]).tolist(),
            'min': float(values.min()), 'max': float(values.max())}


def summarize(root):
    rows = read(root/'part2a/results.json', [])
    manifest = read(root/'part2a/run_manifest.json')
    rng = np.random.default_rng(20261003)
    summary = {'scope': 'Same-state CE vs Steer, fresh AdamW moments, independent diagnostic batches',
               'uncertainty': 'Batch bootstrap only; fixed probe examples, one training seed. No trajectory reconstruction.',
               'methods': {}, 'matched_checkpoint_contrasts': {}, 'completed_utc': utc()}
    metrics = ('gradient_cosine_post_clip', 'native_adamw_update_cosine', 'effective_common_update_cosine')
    probes = ('dev_nll', 'base_confident_probability', 'boolq_chat_yes_no',
              'boolq_chat_gold_no_yes_no', 'boolq_chat_accuracy')
    by_model = {}
    for model in manifest['models']:
        selected = sorted([r for r in rows if r['model'] == model], key=lambda r: r['batch'])
        if {r['batch'] for r in selected} != set(range(manifest['batches'])):
            raise ValueError(f'Incomplete or unbalanced batches for {model}')
        by_model[model] = {r['batch']: r for r in selected}
        result = {key: interval([r[key] for r in selected], rng) for key in metrics}
        result['steer_minus_ce_probe_delta'] = {
            key: interval([r['trials']['steer']['probe_delta'][key] - r['trials']['ce']['probe_delta'][key]
                           for r in selected], rng) for key in probes}
        result['objectives'] = {objective: {
            'effective_common_update_l2': interval([r['trials'][objective]['effective_common_update_l2'] for r in selected], rng),
            'probe_delta': {key: interval([r['trials'][objective]['probe_delta'][key] for r in selected], rng)
                            for key in probes}} for objective in ('ce', 'steer')}
        result['layers'] = {}
        for key in ('layer_effective_update_cosine', 'layer_native_gradient_cosine', 'layer_native_update_cosine'):
            layers = sorted({layer for r in selected for layer in r.get(key, {})}, key=lambda x: (x == 'other', int(x) if x.isdigit() else 1000))
            result['layers'][key] = {layer: interval([r.get(key, {}).get(layer) for r in selected], rng) for layer in layers}
        summary['methods'][model] = result
    # These compare sensitivity of different trained states, not a shared-state
    # causal Full-vs-LoRA direction cosine. Pair by identical diagnostic batch.
    for left, right in (('full_sft', 'lora'), ('steer_full', 'steer_lora'),
                        ('full_sft', 'steer_full'), ('lora', 'steer_lora')):
        if left not in by_model or right not in by_model:
            continue
        summary['matched_checkpoint_contrasts'][f'{left}_minus_{right}'] = {
            key: interval([
                (by_model[left][i]['trials']['steer']['probe_delta'][key] - by_model[left][i]['trials']['ce']['probe_delta'][key]) -
                (by_model[right][i]['trials']['steer']['probe_delta'][key] - by_model[right][i]['trials']['ce']['probe_delta'][key])
                for i in range(manifest['batches'])], rng) for key in probes}
    dump(root/'part2a/BATCH_ANALYSIS.json', summary)
    def fmt(item):
        if item['mean'] is None:
            return 'undefined'
        return f"{item['mean']:.5f} [{item['ci95'][0]:.5f}, {item['ci95'][1]:.5f}]"
    lines = ['# Local update direction results', '', summary['scope'], '', summary['uncertainty'], '',
             'Means and 95% batch-bootstrap intervals. Positive Steer-minus-CE dev NLL means worse local SAMSum likelihood; positive gold-no logodds means more yes bias on no answers.', '',
             '| Checkpoint | Batches | Gradient cosine | Effective update cosine | Steer−CE ΔNLL | Steer−CE Δgold-no logodds |',
             '|---|---:|---:|---:|---:|---:|']
    for model, result in summary['methods'].items():
        lines.append(f"| {model} | {result[metrics[0]]['n']} | {fmt(result[metrics[0]])} | {fmt(result[metrics[2]])} | "
                     f"{fmt(result['steer_minus_ce_probe_delta']['dev_nll'])} | {fmt(result['steer_minus_ce_probe_delta']['boolq_chat_gold_no_yes_no'])} |")
    lines += ['', 'Layerwise intervals and paired contrasts between checkpoints are in BATCH_ANALYSIS.json. The four calibration batches lack newly added layer cosines; layer summaries explicitly report their smaller n.', '',
              'Final displacement comparisons and original full benchmark scores are in ../part1/REPORT.md. FAPM is a postprocessing method and has no separate training gradient. L1 is excluded.', '',
              'Different checkpoint states, fixed held-out probe samples, and reset optimizer moments limit causal interpretation. These trials measure local loss/optimizer sensitivity, not cumulative forgetting or historical training directions.', '']
    (root/'part2a/BATCH_ANALYSIS.md').write_text('\n'.join(lines))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--target-hours', type=float, default=7.5)
    parser.add_argument('--max-batches', type=int, default=128)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    root = args.run_root.resolve()
    if args.summarize_only:
        summarize(root)
        return
    if not 7 <= args.target_hours <= 8:
        raise ValueError('This run is authorized for a 7–8 hour target')
    initial = read(root/'part2a/run_manifest.json')
    status = read(root/'RUN_STATUS.json', {})
    started_utc = status.get('part2a_started_utc', initial.get('initial_started_utc', initial['started_utc']))
    start = datetime.fromisoformat(started_utc).timestamp()
    if args.max_batches < initial['batches']:
        raise ValueError('max-batches cannot be smaller than calibration batches')
    observations = read(root/'BUDGET_OBSERVATIONS.json', [])
    seen = {(r['model'], r['batch']) for r in observations}
    state = {'phase': 'calibration_running', 'target_hours': args.target_hours,
             'original_part2a_started_utc': started_utc,
             'supervisor_pid': __import__('os').getpid(), 'started_utc': utc(),
             'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    dump(root/'BUDGET_STATUS.json', state)
    while True:
        rows = read(root/'part2a/results.json', [])
        for row in rows:
            key = (row['model'], row['batch'])
            if key in seen:
                continue
            # Every completed trial saves its final chat probe before pair
            # reduction. Differences include the previous pair reduction;
            # recover timing even if the supervisor starts after that model.
            probe = root/'part2a'/f'{key[0]}-steer.step{key[1]+1}.boolq-chat.json'
            observations.append({'model': key[0], 'batch': key[1],
                                 'timestamp': probe.stat().st_mtime,
                                 'timestamp_source': probe.name})
            seen.add(key)
        dump(root/'BUDGET_OBSERVATIONS.json', observations)
        status = read(root/'RUN_STATUS.json', {})
        exit_file = root/'logs/exit-status.txt'
        if exit_file.exists() and status.get('finished_utc'):
            if int(exit_file.read_text()) != 0 or status.get('part2a') != 'complete':
                raise RuntimeError('Calibration failed; extension will not run')
            break
        time.sleep(15)
    per_model = {}
    for model in initial['models']:
        recorded = sorted([r for r in observations if r['model'] == model], key=lambda r: r['batch'])
        durations = [b['timestamp']-a['timestamp'] for a, b in zip(recorded, recorded[1:])
                     if b['batch'] == a['batch']+1]
        if not durations:
            raise RuntimeError(f'Missing observed calibration timing for {model}')
        per_model[model] = statistics.mean(durations)
    elapsed = time.time()-start
    batches, setup = choose_batches(elapsed, args.target_hours*3600,
                                    sum(per_model.values()), initial['batches'], args.max_batches)
    command = [sys.executable, '-u', str(ROOT/'eval/probe_checkpoint_updates.py'),
               '--output-dir', str(root/'part2a'), '--resume', '--batches', str(batches),
               '--accumulation', str(initial['accumulation']), '--dev-samples', str(len(initial['dev_indices'])),
               '--boolq-samples', str(len(initial['boolq_doc_ids'])), '--seed', str(initial['seed']),
               '--learning-rate', str(initial['learning_rate']), '--models', *initial['models']]
    state.update({'phase': 'extension_running', 'calibration_elapsed_seconds': elapsed,
                  'measured_pair_seconds_by_model': per_model, 'estimated_repeat_setup_seconds': setup,
                  'final_batches_per_model': batches, 'unique_train_examples': batches*initial['accumulation'],
                  'total_objective_steps': batches*len(initial['models'])*2,
                  'estimated_total_hours': (elapsed+setup+(batches-initial['batches'])*sum(per_model.values())*1.05)/3600,
                  'command': command, 'calibrated_utc': utc()})
    current_batches = initial['batches']
    state['extension_stages'] = []
    while batches > current_batches:
        command[command.index('--batches')+1] = str(batches)
        state.update({'command': command, 'final_batches_per_model': batches,
                      'unique_train_examples': batches*initial['accumulation'],
                      'total_objective_steps': batches*len(initial['models'])*2})
        stage = {'started_utc': utc(), 'from_batches': current_batches, 'to_batches': batches}
        state['extension_stages'].append(stage)
        with (root/'logs/part2a-extension.log').open('a') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state['extension_pid'] = process.pid
            dump(root/'BUDGET_STATUS.json', state)
            status.update({'part2a': 'extension_running', 'budget_target_hours': args.target_hours,
                           'final_batches_per_model': batches, 'extension_pid': process.pid,
                           'part2a_started_utc': started_utc})
            dump(root/'RUN_STATUS.json', status)
            code = process.wait()
            state['extension_exit_code'] = code
            if code:
                raise RuntimeError(f'Extension exited with code {code}')
        stage['finished_utc'] = utc()
        stage['exit_code'] = code
        previous_batches, current_batches = current_batches, batches
        elapsed = time.time()-start
        if elapsed >= 7*3600 or current_batches >= args.max_batches:
            break
        # If the new implementation ran faster, use its measured durations
        # to add meaningful balanced batches rather than leaving a short run.
        expanded = read(root/'part2a/results.json')
        round_seconds = sum(statistics.mean([
            r['pair_elapsed_seconds'] for r in expanded
            if r['model'] == model and r['batch'] >= previous_batches
        ]) for model in initial['models'])
        remaining = args.target_hours*3600-elapsed-setup
        additional = max(0, math.floor(remaining/(round_seconds*1.05)))
        batches = min(args.max_batches, current_batches+additional)
        state['updated_round_seconds'] = round_seconds
        state['estimated_total_hours'] = (elapsed+setup+additional*round_seconds*1.05)/3600
        dump(root/'BUDGET_STATUS.json', state)
    batches = current_batches
    summarize(root)
    state.update({'phase': 'complete', 'finished_utc': utc(), 'actual_total_hours': (time.time()-start)/3600})
    dump(root/'BUDGET_STATUS.json', state)
    status.update({'part2a': 'complete', 'finished_utc': utc(), 'budget_target_hours': args.target_hours,
                   'final_batches_per_model': batches, 'actual_total_hours': state['actual_total_hours']})
    dump(root/'RUN_STATUS.json', status)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Preserve the exception in the detached log. Existing rows remain usable.
        for i, argument in enumerate(sys.argv):
            if argument == '--run-root' and i+1 < len(sys.argv):
                root = Path(sys.argv[i+1])
                state = read(root/'BUDGET_STATUS.json', {})
                state.update({'phase': 'failed', 'error': str(error), 'failed_utc': utc()})
                dump(root/'BUDGET_STATUS.json', state)
                status = read(root/'RUN_STATUS.json', {})
                status.update({'part2a': 'budget_extension_failed', 'budget_error': str(error)})
                dump(root/'RUN_STATUS.json', status)
        raise
