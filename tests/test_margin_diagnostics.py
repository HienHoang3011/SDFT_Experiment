import math
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from margin_diagnostics import boost_margin, margin_metrics


class MarginDiagnosticsTests(unittest.TestCase):
    def test_direct_margin_preserves_relative_off_target_distribution(self):
        p = torch.tensor([[0.05, 0.7, 0.2, 0.05]]).log()
        target = torch.tensor([0])
        q, gap, boost, _ = boost_margin(p, target, math.log(2.5))
        self.assertGreater(boost.item(), 0)
        metrics = margin_metrics(p, q, target)
        self.assertLess(metrics['off_target_kl_base_to_model'].item(), 1e-6)
        self.assertLess(metrics['one_step_base_margin_target_to_model_kl'].item(), 1e-6)
        self.assertAlmostEqual(metrics['model_gt_margin_gap'].item(), math.log(2.5), places=6)

    def test_satisfied_margin_can_have_arbitrarily_diffuse_probability(self):
        # Both outputs meet the same margin; neither gets any steering gradient.
        sharp = torch.tensor([[0., -10., -10., -10.]]).log_softmax(-1)
        diffuse = torch.tensor([[math.log(2.5), 0., 0., 0.]]).log_softmax(-1)
        for logp in (sharp, diffuse):
            target, _, _, _ = boost_margin(logp, torch.tensor([0]), math.log(2.5))
            torch.testing.assert_close(target, logp)
        diagnostics = margin_metrics(sharp, diffuse, torch.tensor([0]))
        self.assertGreater(diagnostics['off_target_model_entropy'].item(), 1.)
        self.assertLess(diffuse.exp()[0, 0].item(), 0.5)

    def test_identical_reference_has_no_off_target_drift(self):
        p = torch.tensor([[0.8, 0.15, 0.05]]).log()
        metrics = margin_metrics(p, p, torch.tensor([0]))
        self.assertEqual(metrics['off_target_kl_base_to_model'].item(), 0)
        self.assertEqual(metrics['model_self_margin_kl'].item(), 0)


if __name__ == '__main__':
    unittest.main()
