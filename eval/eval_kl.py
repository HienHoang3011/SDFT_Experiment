"""Compare full-vocabulary softmax distributions on identical teacher-forced prefixes."""
import argparse
import gc
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'methods'))
from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens
from distribution_metrics import token_metrics, summarize
from margin_diagnostics import margin_metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--data-root', default='data/samsum_emnlp2019')
    parser.add_argument('--probe-jsonl', type=Path, help='Optional held-out JSONL with messages including reference assistant answers')
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model-path', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--tokens-per-sample', type=int, default=32)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--margin-diagnostics', action='store_true')
    parser.add_argument('--margin-x-factor', type=float, default=1.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--selection', choices=['first', 'random'], default='first')
    args = parser.parse_args()
    if args.samples <= 0 or args.tokens_per_sample <= 0 or args.top_k <= 0 or not 0 < args.temperature < float('inf'):
        parser.error('samples, tokens-per-sample, top-k and finite temperature must be positive')
    return args


def load_model(path, base_path, tokenizer):
    adapter = (Path(path) / 'adapter_config.json').is_file()
    model = AutoModelForCausalLM.from_pretrained(base_path if adapter else path, torch_dtype=torch.bfloat16, device_map='cuda')
    if adapter:
        config = json.loads((Path(path) / 'adapter_config.json').read_text())
        if config.get('base_model_name_or_path', '').rstrip('/').split('/')[-1] != str(base_path).rstrip('/').split('/')[-1]:
            raise ValueError(f'Adapter base differs from reference: {path}')
        model = PeftModel.from_pretrained(model, path)
    sync_model_special_tokens(model, tokenizer)
    return model.eval()


def make_probes(dataset, tokenizer, count, max_tokens, selection='first', seed=42):
    indices = list(range(len(dataset)))
    if selection == 'random':
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    probes = []
    for index in indices[:count]:
        encoded = tokenizer.apply_chat_template(dataset[index]['messages'], tokenize=True, add_generation_prompt=False, return_dict=True, return_assistant_tokens_mask=True)
        ids, mask = encoded['input_ids'], encoded['assistant_masks']
        positions = [i - 1 for i in range(1, len(ids)) if mask[i]]
        if len(positions) > max_tokens:
            selected = torch.linspace(0, len(positions) - 1, max_tokens).round().long()
            positions = [positions[i] for i in selected.tolist()]
        if positions:
            probes.append((index, torch.tensor([ids], dtype=torch.long), positions))
    if not probes:
        raise ValueError('No assistant tokens in selected probes')
    return probes


@torch.inference_mode()
def distributions(model, probes, temperature=1.0):
    for _, ids, positions in probes:
        outputs = model(input_ids=ids.to('cuda'), use_cache=False)
        logits = outputs.logits[0, positions].float() / temperature
        yield F.log_softmax(logits, dim=-1).cpu()
        del outputs, logits


def evaluate(name, path, base_path, tokenizer, probes, reference, args):
    model = None if name == 'base' else load_model(path, base_path, tokenizer)
    current = iter(reference) if model is None else distributions(model, probes, args.temperature)
    buckets = {}
    lengths = []
    with (args.output_dir / f'{name}.tokens.jsonl').open('w') as handle:
        for (example, ids, positions), base_cpu, ft_cpu in zip(probes, reference, current):
            base, ft = base_cpu.to('cuda'), ft_cpu.to('cuda')
            targets = ids[0, torch.tensor(positions) + 1]
            metrics = token_metrics(base, ft, targets, args.top_k)
            if args.margin_diagnostics:
                metrics.update(margin_metrics(base, ft, targets, args.margin_x_factor))
            lengths.append(len(positions))
            for key, values in metrics.items():
                buckets.setdefault(key, []).append(values.cpu())
            # Show both probabilities on the SAME union of top-k tokens and target.
            p, q = base.exp().cpu(), ft.exp().cpu()
            base_top, ft_top = base.topk(args.top_k, -1).indices.cpu(), ft.topk(args.top_k, -1).indices.cpu()
            for row, position in enumerate(positions):
                union = sorted(set(base_top[row].tolist() + ft_top[row].tolist() + [targets[row].item()]))
                record = {'example_id': example, 'prediction_position': position, 'target_token_id': targets[row].item(), 'metrics': {key: values[row].item() for key, values in metrics.items()}, 'tokens': [{'id': token, 'text': tokenizer.decode([token]), 'base_probability': p[row, token].item(), 'model_probability': q[row, token].item()} for token in union]}
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
    if model is not None:
        del current, model
        gc.collect()
        torch.cuda.empty_cache()
    stats = summarize({key: torch.cat(values) for key, values in buckets.items()}, lengths, args.seed)
    result = {'model': name, 'model_path': path, 'assistant_tokens': sum(lengths), 'dev_examples': len(probes), 'statistics': stats, **{key: value['mean'] for key, value in stats.items()}}
    (args.output_dir / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'statistics'}), flush=True)
    return result


def main():
    args = parse_args()
    specifications = [('base', args.base_model)]
    for spec in args.model_path:
        name, separator, path = spec.partition('=')
        if not separator or not re.fullmatch(r'[A-Za-z0-9_-]+', name) or not path:
            raise ValueError('--model-path must be NAME=PATH with a safe filename name')
        if name == 'base':
            if Path(path).resolve() != Path(args.base_model).resolve():
                raise ValueError('The reserved base name must refer to --base-model')
            continue
        if name in [item[0] for item in specifications]:
            raise ValueError(f'Duplicate model name: {name}')
        specifications.append((name, path))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    configure_qwen2_base_tokenizer(tokenizer, args.base_model)
    dataset = [json.loads(line) for line in args.probe_jsonl.read_text().splitlines() if line.strip()] if args.probe_jsonl else Dataset.load_from_disk(str(Path(args.data_root) / 'dev'))
    probes = make_probes(dataset, tokenizer, args.samples, args.tokens_per_sample, args.selection, args.seed)
    base = load_model(args.base_model, args.base_model, tokenizer)
    reference = list(distributions(base, probes, args.temperature))
    del base
    gc.collect()
    torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [evaluate(name, path, args.base_model, tokenizer, probes, reference, args) for name, path in specifications]
    fingerprint = hashlib.sha256(json.dumps([(index, ids.tolist(), positions) for index, ids, positions in probes]).encode()).hexdigest()
    report = {'dataset': str(args.probe_jsonl) if args.probe_jsonl else 'SAMSum dev', 'base_model': args.base_model, 'selection': args.selection, 'seed': args.seed, 'samples_requested': args.samples, 'tokens_per_sample': args.tokens_per_sample, 'temperature': args.temperature, 'top_k': args.top_k, 'margin_x_factor': args.margin_x_factor if args.margin_diagnostics else None, 'probe_sha256': fingerprint, 'precision': 'bf16 model forward; fp32 full-vocabulary log-softmax and metrics; fp32 reference storage', 'units': 'nats per assistant token, full vocabulary', 'conditioning': 'same teacher-forced reference prefix for every model', 'ci': '95% percentile bootstrap, 1000 resamples of examples; conditional on selected probes', 'rows': rows}
    (args.output_dir / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    lines = ['# Distribution comparison', '', f"Dataset: {report['dataset']}; {len(probes)} examples; {rows[0]['assistant_tokens']} tokens; T={args.temperature}.", '', '| Model | KL(base ∥ model) | KL(model ∥ base) | JS | TV | Top-1 agreement | Top-k overlap | Target NLL |', '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [row[key] for key in ('kl_base_to_model_nats_per_token', 'kl_model_to_base_nats_per_token', 'js_nats_per_token', 'total_variation', 'top1_agreement', 'topk_overlap', 'model_target_nll_nats')]
        lines.append('| ' + row['model'] + ' | ' + ' | '.join(f'{value:.6f}' for value in values) + ' |')
    lines += ['', 'Lower divergence means closer to base, not necessarily better task performance. Target NLL measures reference-answer fit. SAMSum drift alone does not establish general capability forgetting.', '', 'JSON includes token percentiles and example-bootstrap confidence intervals. Token JSONL shows base/model probabilities on a common union of top-k tokens and the reference token; metrics always use the full vocabulary.']
    (args.output_dir / 'comparison.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
