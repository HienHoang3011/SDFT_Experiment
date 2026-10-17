"""Exact zero-shot BoolQ answer logits and full-vocabulary margin diagnostics."""
import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
import torch
from transformers import AutoTokenizer
from eval_kl import load_model
from distribution_metrics import token_metrics, summarize
from margin_diagnostics import margin_metrics
from answer_metrics import binary_answer_metrics


@torch.inference_mode()
def get_distributions(model, documents, tokenizer, batch_size, prompt_format):
    for start in range(0, len(documents), batch_size):
        batch = documents[start:start + batch_size]
        prompts = [f"{row['passage']}\nQuestion: {row['question']}?\nAnswer:" for row in batch]
        if prompt_format == 'chat':
            prompts = [tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True) for prompt in prompts]
        encoded = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors='pt').to('cuda')
        outputs = model(**encoded, use_cache=False, logits_to_keep=1)
        yield outputs.logits[:, -1].float().log_softmax(-1).cpu()
        del outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--documents', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--model-path', action='append', required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--prompt-format', choices=['raw', 'chat'], default='chat')
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('batch-size must be positive')
    for specification in args.model_path:
        name, separator, path = specification.partition('=')
        if not separator or not path or not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            parser.error('--model-path must be safe NAME=PATH')
    documents = [json.loads(line) for line in args.documents.read_text().splitlines()]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = '<|endoftext|>'
    tokenizer.padding_side = 'left'
    # lm-eval removes target_delimiter for ChatML without a gen_prefix. This diagnostic
    # scores only the single no/yes token; any token lm-eval adds after the answer is
    # not scored here and need not have equal probability for both choices.
    delimiter = ' ' if args.prompt_format == 'raw' else ''
    choices = [tokenizer.encode(delimiter + answer, add_special_tokens=False) for answer in ('no', 'yes')]
    if any(len(choice) != 1 for choice in choices):
        raise ValueError('BoolQ choices must be single tokens for this exact scorer')
    choice_ids = torch.tensor([choice[0] for choice in choices])
    labels = torch.tensor([doc['label'] for doc in documents])
    targets = choice_ids[labels]
    base = load_model(args.base_model, args.base_model, tokenizer)
    reference = list(get_distributions(base, documents, tokenizer, args.batch_size, args.prompt_format))
    base_predictions = torch.cat([value[:, choice_ids].argmax(-1) for value in reference])
    del base
    gc.collect()
    torch.cuda.empty_cache()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specifications = [('base', args.base_model)] + [tuple(spec.split('=', 1)) for spec in args.model_path if not spec.startswith('base=')]
    rows = []
    for name, path in specifications:
        model = None if name == 'base' else load_model(path, args.base_model, tokenizer)
        current = iter(reference) if model is None else get_distributions(model, documents, tokenizer, args.batch_size, args.prompt_format)
        buckets = {}
        predictions = []
        offset = 0
        with (args.output_dir / f'{name}.examples.jsonl').open('w') as handle:
            for p, q in zip(reference, current):
                count = len(p)
                batch_targets = targets[offset:offset + count]
                batch_labels = labels[offset:offset + count]
                metrics = token_metrics(p.to('cuda'), q.to('cuda'), batch_targets)
                metrics.update(margin_metrics(p.to('cuda'), q.to('cuda'), batch_targets))
                base_score, model_score = p[:, choice_ids], q[:, choice_ids]
                predicted = model_score.argmax(-1)
                predictions.append(predicted)
                metrics.update({
                    'model_yes_minus_no_logit': model_score[:, 1] - model_score[:, 0],
                    'base_yes_minus_no_logit': base_score[:, 1] - base_score[:, 0],
                    'model_yes_probability_among_choices': model_score.softmax(-1)[:, 1],
                    'base_yes_probability_among_choices': base_score.softmax(-1)[:, 1],
                    'correct_answer_logodds_shift': (model_score[:, 1] - model_score[:, 0] - base_score[:, 1] + base_score[:, 0]) * (2 * batch_labels - 1),
                })
                for key, values in metrics.items():
                    buckets.setdefault(key, []).append(values.cpu())
                for j in range(count):
                    record = {'doc_id': documents[offset+j]['doc_id'], 'label': batch_labels[j].item(), 'base_prediction': base_predictions[offset+j].item(), 'model_prediction': predicted[j].item(), 'base_choice_logprobs': base_score[j].tolist(), 'model_choice_logprobs': model_score[j].tolist(), 'metrics': {key: value[j].item() for key, value in metrics.items()}}
                    handle.write(json.dumps(record) + '\n')
                offset += count
        predicted = torch.cat(predictions)
        correct = predicted.eq(labels)
        base_correct = base_predictions.eq(labels)
        all_metrics = {key: torch.cat(parts) for key, parts in buckets.items()}
        statistics = summarize(all_metrics, [1] * len(documents))
        groups = {}
        for group, mask in {'gold_no': labels.eq(0), 'gold_yes': labels.eq(1), 'base_correct_model_wrong': base_correct & ~correct, 'base_wrong_model_correct': ~base_correct & correct, 'both_correct': base_correct & correct}.items():
            groups[group] = {'examples': mask.sum().item(), 'means': {key: value[mask].mean().item() for key, value in all_metrics.items()} if mask.any() else {}}
        row = {'model': name, 'model_path': path, 'examples': len(documents), 'accuracy': correct.float().mean().item(), 'predicted_yes_fraction': predicted.float().mean().item(), 'base_correct_model_wrong': (base_correct & ~correct).sum().item(), 'base_wrong_model_correct': (~base_correct & correct).sum().item(), 'statistics': statistics, 'groups': groups, **{key: value['mean'] for key, value in statistics.items()}}
        row.update(binary_answer_metrics(all_metrics['model_yes_minus_no_logit'], labels, predicted))
        rows.append(row)
        (args.output_dir / f'{name}.json').write_text(json.dumps(row, indent=2) + '\n')
        print(json.dumps({key: row[key] for key in ('model', 'accuracy', 'predicted_yes_fraction', 'base_correct_model_wrong', 'base_wrong_model_correct', 'model_entropy_nats', 'off_target_kl_base_to_model')}), flush=True)
        if model is not None:
            del model, current
            gc.collect()
            torch.cuda.empty_cache()
    summary = {'dataset': 'aps/super_glue boolq validation', 'samples_sha256': hashlib.sha256(args.documents.read_bytes()).hexdigest(), 'conditioning': 'Zero-shot BoolQ question; answer delimiter is a space for raw and empty for ChatML; only the single answer token is scored, not any trailing token', 'choice_delimiter': delimiter, 'prompt_format': args.prompt_format, 'benchmark_match': args.prompt_format == 'chat', 'batch_size': args.batch_size, 'choice_token_ids': choice_ids.tolist(), 'margin_x_factor': 1.5, 'rows': rows}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    lines = ['# BoolQ margin diagnostics', '', f'{len(documents)} held-out examples; {args.prompt_format} prompts. ChatML matches the saved benchmark; raw prompts are a formatting control. Choice scores are full-vocabulary log probabilities, without generation.', '', '| Model | Accuracy % | Predicted yes % | Base-correct → wrong | Base-wrong → correct | KL(base ∥ model) | KL(model ∥ base) | Entropy | Off-target KL(base ∥ model) |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| {row['model']} | {100*row['accuracy']:.3f} | {100*row['predicted_yes_fraction']:.3f} | {row['base_correct_model_wrong']} | {row['base_wrong_model_correct']} | {row['kl_base_to_model_nats_per_token']:.6f} | {row['kl_model_to_base_nats_per_token']:.6f} | {row['model_entropy_nats']:.6f} | {row['off_target_kl_base_to_model']:.6f} |")
    lines += ['', 'Off-target KL renormalizes the vocabulary after excluding the reference answer token. It diagnoses changes beyond a direct ground-truth logit boost. This is a checkpoint comparison, not a causal ablation.']
    (args.output_dir / 'comparison.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
