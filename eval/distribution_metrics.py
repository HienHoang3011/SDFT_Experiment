"""Full-vocabulary distribution metrics; inputs are log probabilities [tokens, vocab]."""
import math
import torch


def token_metrics(base, current, targets, top_k=10):
    if base.shape != current.shape or base.ndim != 2:
        raise ValueError('Expected matching [tokens, vocabulary] distributions')
    if not 1 <= top_k <= base.shape[-1]:
        raise ValueError('top_k must fit the vocabulary')
    base = base.float().log_softmax(-1)
    current = current.float().log_softmax(-1)
    if not torch.isfinite(base).all() or not torch.isfinite(current).all():
        raise ValueError('Nonfinite log probabilities')
    p, q = base.exp(), current.exp()
    mixture = torch.logaddexp(base, current) - math.log(2)
    base_top = base.topk(top_k, dim=-1).indices
    current_top = current.topk(top_k, dim=-1).indices
    target = targets.to(base.device).reshape(-1, 1)
    return {
        'kl_base_to_model_nats_per_token': (p * (base - current)).sum(-1).clamp_min(0),
        'kl_model_to_base_nats_per_token': (q * (current - base)).sum(-1).clamp_min(0),
        'js_nats_per_token': (0.5 * ((p * (base - mixture)).sum(-1) + (q * (current - mixture)).sum(-1))).clamp_min(0),
        'total_variation': 0.5 * (p - q).abs().sum(-1),
        'top1_agreement': (base.argmax(-1) == current.argmax(-1)).float(),
        'topk_overlap': (base_top.unsqueeze(-1) == current_top.unsqueeze(-2)).any(-1).float().mean(-1),
        'base_entropy_nats': -(p * base).sum(-1),
        'model_entropy_nats': -(q * current).sum(-1),
        'base_target_probability': p.gather(-1, target).squeeze(-1),
        'model_target_probability': q.gather(-1, target).squeeze(-1),
        'base_target_nll_nats': -base.gather(-1, target).squeeze(-1),
        'model_target_nll_nats': -current.gather(-1, target).squeeze(-1),
    }


def summarize(values, example_lengths, seed=42, bootstrap_samples=1000):
    """Token mean and percentiles; CI resamples examples, preserving their tokens."""
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.tensor(example_lengths, dtype=torch.float64)
    indices = torch.randint(len(lengths), (bootstrap_samples, len(lengths)), generator=generator)
    result = {}
    for name, tensor in values.items():
        tensor = tensor.double().cpu()
        sums = torch.stack([part.sum() for part in tensor.split(example_lengths)])
        means = sums[indices].sum(-1) / lengths[indices].sum(-1)
        result[name] = {
            'mean': tensor.mean().item(),
            'p50': tensor.quantile(0.5).item(),
            'p95': tensor.quantile(0.95).item(),
            'max': tensor.max().item(),
            'example_bootstrap_ci95': means.quantile(torch.tensor([0.025, 0.975], dtype=torch.float64)).tolist(),
        }
    return result
