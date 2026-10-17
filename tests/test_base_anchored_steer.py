import contextlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'methods'))
from base_anchored_steer import BaseAnchoredLoss, capture_base_log_probs
from train_steer import SteeredLoss


class TinyAdapterModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Parameter(torch.tensor([0.2, 0.5, -0.3]), requires_grad=False)
        self.lora_delta = torch.nn.Parameter(torch.tensor([0.9, -0.5, 0.4]))
        self.dropout = torch.nn.Dropout(0.8)
        self.disabled = False

    @contextlib.contextmanager
    def disable_adapter(self):
        previous = self.disabled
        self.disabled = True
        try:
            yield
        finally:
            self.disabled = previous

    def forward(self, input_ids, use_cache=False):
        logits = self.base if self.disabled else self.base + self.dropout(self.lora_delta)
        return SimpleNamespace(logits=logits.expand(*input_ids.shape, 3))


class AnchorTests(unittest.TestCase):
    def test_teacher_is_frozen_base_and_restores_student(self):
        model = TinyAdapterModel().train()
        inputs = {'input_ids': torch.tensor([[0, 1, 2]]), 'labels': torch.tensor([[-100, -100, 2]])}
        reference = capture_base_log_probs(model, inputs)
        torch.testing.assert_close(reference, model.base.log_softmax(-1).reshape(1, 3))
        self.assertFalse(reference.requires_grad)
        self.assertTrue(model.training)
        self.assertFalse(model.disabled)
        model.lora_delta.data.add_(5)
        torch.testing.assert_close(capture_base_log_probs(model, inputs), reference)

    def test_rejects_trainable_base(self):
        model = TinyAdapterModel()
        model.base.requires_grad_(True)
        with self.assertRaises(ValueError):
            capture_base_log_probs(model, {'input_ids': torch.tensor([[0, 1]]), 'labels': torch.tensor([[-100, 1]])})

    def test_combined_loss_and_gradients_match_analytic_anchor(self):
        logits = torch.tensor([[[0.1, 0.8, 0.3], [0.5, 0.2, 0.1], [0., 0., 0.]]], requires_grad=True)
        labels = torch.tensor([[-100, 2, -100]])
        outputs = SimpleNamespace(logits=logits)
        base = torch.tensor([[0.7, 0.2, 0.1]]).log()
        steering = SteeredLoss(1.5)
        anchor = BaseAnchoredLoss(steering, 0.8)
        anchor.reference_log_probs = base
        loss = anchor(outputs, labels, num_items_in_batch=4)
        expected = steering(outputs, labels, 4) + 0.8 * F.kl_div(logits[0, :1].log_softmax(-1), base, log_target=True, reduction='sum') / 4
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(logits.grad[0, 0].abs().sum().item(), 0)
        self.assertEqual(logits.grad[0, 1:].abs().sum().item(), 0)

    def test_steer_gradient_is_adaptively_scaled_cross_entropy(self):
        import math
        logits = torch.tensor([[[0.1, 1.0, 0.4], [0., 0., 0.]]], requires_grad=True)
        labels = torch.tensor([[-100, 0]])
        loss = SteeredLoss(1.5)(SimpleNamespace(logits=logits), labels)
        gradient = torch.autograd.grad(loss, logits)[0][0, 0]
        p = logits[0, 0].detach().softmax(-1)
        max_other = logits[0, 0, 1:].detach().max()
        boost = (max_other + math.log(2.5) - logits[0, 0, 0].detach()).clamp_min(0)
        z = 1 + (boost.exp() - 1) * p[0]
        scale = 1 - 1/z
        ce_gradient = p.clone()
        ce_gradient[0] -= 1
        torch.testing.assert_close(gradient, scale * ce_gradient)

    def test_zero_at_base_and_fully_masked_batch(self):
        logits = torch.tensor([[[0.8, 0.2], [0.1, 0.4]]], requires_grad=True)
        outputs = SimpleNamespace(logits=logits)
        anchor = BaseAnchoredLoss(SteeredLoss(1.5), 1)
        anchor.reference_log_probs = logits[0, :1].detach().log_softmax(-1)
        labels = torch.tensor([[-100, 0]])
        torch.testing.assert_close(anchor(outputs, labels), anchor.steered_loss(outputs, labels))
        masked = anchor(outputs, torch.full_like(labels, -100))
        self.assertEqual(masked.item(), 0)
        masked.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
