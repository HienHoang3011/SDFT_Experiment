import math
import sys
import unittest
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'eval'))
from distribution_metrics import token_metrics, summarize


class DistributionMetricTests(unittest.TestCase):
    def test_identical_distributions(self):
        logp = torch.tensor([[0.8, 0.15, 0.05], [0.1, 0.2, 0.7]]).log()
        result = token_metrics(logp, logp, torch.tensor([0, 2]), top_k=2)
        for key in ('kl_base_to_model_nats_per_token', 'kl_model_to_base_nats_per_token', 'js_nats_per_token', 'total_variation'):
            torch.testing.assert_close(result[key], torch.zeros(2), atol=1e-7, rtol=0)
        for key in ('top1_agreement', 'topk_overlap'):
            torch.testing.assert_close(result[key], torch.ones(2))

    def test_known_binary_distributions_and_direction(self):
        p, q = torch.tensor([[0.9, 0.1]]), torch.tensor([[0.5, 0.5]])
        values = token_metrics(p.log(), q.log(), torch.tensor([0]), top_k=2)
        self.assertAlmostEqual(values['kl_base_to_model_nats_per_token'].item(), 0.9 * math.log(1.8) + 0.1 * math.log(0.2), places=6)
        self.assertAlmostEqual(values['kl_model_to_base_nats_per_token'].item(), 0.5 * math.log(0.5/0.9) + 0.5 * math.log(5), places=6)
        self.assertAlmostEqual(values['total_variation'].item(), 0.4, places=6)
        self.assertAlmostEqual(values['model_target_nll_nats'].item(), math.log(2), places=6)
        reverse = token_metrics(q.log(), p.log(), torch.tensor([0]), top_k=2)
        torch.testing.assert_close(values['js_nats_per_token'], reverse['js_nats_per_token'])
        self.assertLessEqual(values['js_nats_per_token'].item(), math.log(2))

    def test_topk_overlap_uses_token_sets(self):
        values = token_metrics(torch.tensor([[5., 4., 1., 0.]]), torch.tensor([[0., 5., 4., 1.]]), torch.tensor([2]), top_k=2)
        self.assertEqual(values['topk_overlap'].item(), 0.5)
        self.assertEqual(values['top1_agreement'].item(), 0)

    def test_example_bootstrap_and_token_weighting(self):
        values = {'metric': torch.tensor([0., 2., 2.])}
        summary = summarize(values, [1, 2])['metric']
        self.assertAlmostEqual(summary['mean'], 4/3)
        self.assertEqual(summary['example_bootstrap_ci95'], [0., 2.])
        self.assertEqual(summarize(values, [1, 2]), summarize(values, [1, 2]))

    def test_rejects_shape_and_nonfinite_inputs(self):
        with self.assertRaises(ValueError):
            token_metrics(torch.zeros(1, 3), torch.zeros(2, 3), torch.tensor([0]))
        with self.assertRaises(ValueError):
            token_metrics(torch.tensor([[float('nan'), 0.]]), torch.zeros(1, 2), torch.tensor([0]), top_k=1)


if __name__ == '__main__':
    unittest.main()
