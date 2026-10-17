"""Steer plus a frozen-base KL anchor, sharing frozen weights with a LoRA student."""
import math
import torch
import torch.nn.functional as F
from trl import SFTTrainer


def capture_base_log_probs(model, inputs):
    """Adapter-disabled, dropout-free teacher forward before the student graph."""
    if not hasattr(model, 'disable_adapter'):
        raise ValueError('The shared frozen-base teacher requires a PEFT model')
    if any(parameter.requires_grad and 'lora_' not in name for name, parameter in model.named_parameters()):
        raise ValueError('Base weights must stay frozen for adapter-disabled reference inference')
    active = inputs['labels'][..., 1:].reshape(-1).ne(-100)
    model_inputs = {key: value for key, value in inputs.items() if key not in {'labels', 'shift_labels', '_prediction_loss_only'}}
    model_inputs['use_cache'] = False
    was_training = model.training
    try:
        model.eval()
        with model.disable_adapter(), torch.no_grad():
            outputs = model(**model_inputs)
            logits = outputs.logits[..., :-1, :]
            selected = logits.reshape(-1, logits.shape[-1])[active].float()
            return F.log_softmax(selected, dim=-1).detach()
    finally:
        model.train(was_training)


class BaseAnchoredLoss:
    def __init__(self, steered_loss, weight):
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError('base KL weight must be finite and positive')
        self.steered_loss = steered_loss
        self.weight = weight
        self.reference_log_probs = None

    def bind_model(self, model):
        self.steered_loss.bind_model(model)

    def __call__(self, outputs, labels, num_items_in_batch=None):
        steering_loss = self.steered_loss(outputs, labels, num_items_in_batch)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        shifted = logits[..., :-1, :]
        active = labels[..., 1:].reshape(-1).ne(-100)
        if not active.any():
            return steering_loss
        current = shifted.reshape(-1, shifted.shape[-1])[active].float().log_softmax(-1)
        if self.reference_log_probs is None or self.reference_log_probs.shape != current.shape:
            raise RuntimeError('Reference probabilities must match every active assistant token')
        kl_sum = F.kl_div(current, self.reference_log_probs, log_target=True, reduction='sum')
        denominator = active.sum() if num_items_in_batch is None else num_items_in_batch
        denominator = torch.as_tensor(denominator, device=current.device, dtype=current.dtype).clamp_min(1)
        accumulator = self.steered_loss.metric_accumulator
        if accumulator is not None:
            mode = 'train' if self.steered_loss.model.training else 'eval'
            accumulator.add(mode, 'base_anchor_kl', kl_sum, active.sum())
        return steering_loss + self.weight * kl_sum / denominator


class BaseAnchoredSFTTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        objective = self.compute_loss_func
        if not isinstance(objective, BaseAnchoredLoss):
            raise TypeError('BaseAnchoredSFTTrainer requires BaseAnchoredLoss')
        # The teacher is the original frozen base with LoRA disabled. This
        # avoids a second 7B copy and forbids trainable base parameters.
        reference_model = self.accelerator.unwrap_model(model)
        with self.compute_loss_context_manager():
            objective.reference_log_probs = capture_base_log_probs(reference_model, inputs)
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            objective.reference_log_probs = None
