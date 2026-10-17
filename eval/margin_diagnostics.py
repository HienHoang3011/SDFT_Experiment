"""Diagnose margin satisfaction, vocabulary-tail drift, and a direct-margin counterfactual."""
import math
import torch


def boost_margin(log_probs, targets, margin):
    rows = torch.arange(len(targets), device=log_probs.device)
    other = log_probs.clone()
    other[rows, targets] = -torch.inf
    gap = log_probs[rows, targets] - other.max(-1).values
    boost = (margin - gap).clamp_min(0)
    target = log_probs.clone()
    target[rows, targets] += boost
    return target.log_softmax(-1), gap, boost, other


def margin_metrics(base, model, targets, x_factor=1.5):
    if not math.isfinite(x_factor) or x_factor <= -1:
        raise ValueError('x_factor must be finite and greater than -1')
    base, model = base.float().log_softmax(-1), model.float().log_softmax(-1)
    targets = targets.to(base.device).reshape(-1)
    margin = math.log1p(x_factor)
    base_target, base_gap, base_boost, base_other = boost_margin(base, targets, margin)
    model_target, model_gap, model_boost, model_other = boost_margin(model, targets, margin)
    # Normalize ONLY non-reference tokens to isolate their relative distribution.
    base_cond = base_other - base_other.logsumexp(-1, keepdim=True)
    model_cond = model_other - model_other.logsumexp(-1, keepdim=True)
    rows = torch.arange(len(targets), device=base.device)
    base_cond[rows, targets] = 0
    model_cond[rows, targets] = 0
    p_other, q_other = base_cond.exp(), model_cond.exp()
    p_other[rows, targets] = 0
    q_other[rows, targets] = 0
    base_logz = torch.logaddexp(base[rows, targets] + base_boost, base_other.logsumexp(-1))
    model_logz = torch.logaddexp(model[rows, targets] + model_boost, model_other.logsumexp(-1))
    return {
        'base_margin_ce_gradient_scale': (-torch.expm1(-base_logz)).clamp_min(0),
        'model_margin_ce_gradient_scale': (-torch.expm1(-model_logz)).clamp_min(0),
        'base_gt_margin_gap': base_gap,
        'model_gt_margin_gap': model_gap,
        'base_margin_active': (base_boost > 0).float(),
        'model_margin_active': (model_boost > 0).float(),
        'base_gt_boost_needed': base_boost,
        'model_gt_boost_needed': model_boost,
        'base_self_margin_kl': (base_target.exp() * (base_target - base)).sum(-1).clamp_min(0),
        'model_self_margin_kl': (model_target.exp() * (model_target - model)).sum(-1).clamp_min(0),
        'off_target_kl_base_to_model': (p_other * (base_cond - model_cond)).sum(-1).clamp_min(0),
        'off_target_kl_model_to_base': (q_other * (model_cond - base_cond)).sum(-1).clamp_min(0),
        'off_target_base_entropy': -(p_other * base_cond).sum(-1),
        'off_target_model_entropy': -(q_other * model_cond).sum(-1),
        'one_step_base_margin_target_to_model_kl': (base_target.exp() * (base_target - model)).sum(-1).clamp_min(0),
        'one_step_base_margin_target_entropy': -(base_target.exp() * base_target).sum(-1),
    }
