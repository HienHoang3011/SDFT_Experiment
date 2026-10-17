"""Weight-space directions, with undefined cosine for zero vectors."""
import math
import torch


def validate_resume(previous, current):
    for key in ('models', 'accumulation', 'dev_indices', 'boolq_doc_ids',
                'seed', 'learning_rate', 'fixed_margin', 'optimizer'):
        if previous[key] != current[key]:
            raise ValueError(f'Resume scope changed: {key}')
    if previous['batches'] > current['batches']:
        raise ValueError('Cannot reduce batch count on resume')
    if previous['train_indices'] != current['train_indices'][:len(previous['train_indices'])]:
        raise ValueError('Resume training batch prefix changed')


def cosine(dot, norm_a_squared, norm_b_squared):
    if norm_a_squared <= 0 or norm_b_squared <= 0:
        return None
    return max(-1., min(1., dot / math.sqrt(norm_a_squared * norm_b_squared)))


def effective_factors(a_before, b_before, a_after, b_after, scale):
    # Exact: dB A0 + B1 dA includes the second-order dB dA term.
    u = torch.cat((b_after - b_before, b_after), dim=1) * scale
    v = torch.cat((a_before, a_after - a_before), dim=0)
    return u, v


def factor_inner(u, v, other_u=None, other_v=None):
    other_u = u if other_u is None else other_u
    other_v = v if other_v is None else other_v
    # Small Gram matrices; fp64 avoids cancellation in mixed factor terms.
    return ((u.double().T @ other_u.double()) *
            (v.double() @ other_v.double().T)).sum().item()


def tensor_inner(a, b, chunk=1048576):
    result = 0.
    for start in range(0, a.numel(), chunk):
        x = a.reshape(-1)[start:start+chunk].double()
        y = b.reshape(-1)[start:start+chunk].double()
        result += (x*y).sum().item()
    return result


def gram_rows(vectors):
    # Tensor chunks make fp64 accumulation feasible for the complete weight space.
    rows = torch.stack([v.reshape(-1) for v in vectors]).double()
    return (rows @ rows.T).cpu()
