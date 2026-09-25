"""CPU-only checks for the fair SAMSum method comparison."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
sys.path.insert(0, str(ROOT / "eval"))

import train_ewc  # noqa: E402
import train_sft  # noqa: E402
import train_sft_lora  # noqa: E402
import train_steer  # noqa: E402
from eval_samsum import extract_summary  # noqa: E402

spec = importlib.util.spec_from_file_location("stable_five_run", ROOT / "benchmarks/stable_five/run.py")
stable_five = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable_five)


class TinyCausalLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(4, 3)
        self.head = torch.nn.Linear(3, 4, bias=False)

    def forward(self, input_ids, labels, use_cache=False):
        logits = self.head(self.embedding(input_ids))
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1))
        return SimpleNamespace(loss=loss)


class TokenizerStub:
    def __call__(self, text, truncation, max_length):
        return {"input_ids": [int(value) for value in text.split()][:max_length]}


class MethodTests(unittest.TestCase):
    def test_empirical_fisher_matches_mean_squared_per_example_gradients(self):
        torch.manual_seed(7)
        model = TinyCausalLM()
        data = [{"text": "0 1 2"}, {"text": "2 3 1"}]
        expected = {name: torch.zeros_like(parameter) for name, parameter in model.named_parameters()}
        for sample in data:
            ids = torch.tensor([[int(value) for value in sample["text"].split()]])
            model.zero_grad(set_to_none=True)
            model(ids, ids).loss.backward()
            for name, parameter in model.named_parameters():
                expected[name] += parameter.grad.detach().square() / len(data)
        fisher, count = train_ewc.estimate_empirical_fisher(model, TokenizerStub(), data, 2, 8, 42)
        self.assertEqual(count, 2)
        for name in expected:
            torch.testing.assert_close(fisher[name], expected[name])
        self.assertFalse(model.training is False)

    def test_ewc_quadratic_value_and_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        loss = train_ewc.EWCQuadraticPenalty.apply(
            parameter, (torch.tensor([1.0, 1.0]),), (torch.tensor([2.0, 3.0]),)
        )
        self.assertAlmostEqual(loss.item(), 14.0)
        loss.backward()
        torch.testing.assert_close(parameter.grad, torch.tensor([4.0, -12.0]))

    def test_steer_loss_is_active_until_margin_is_met(self):
        labels = torch.tensor([[-100, 1]])
        logits = torch.zeros((1, 2, 3), requires_grad=True)
        loss = train_steer.SteeredLoss(1.0)(SimpleNamespace(logits=logits), labels)
        loss.backward()
        self.assertLess(logits.grad[0, 0, 1].item(), 0)
        satisfied = torch.tensor([[[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
        near_zero = train_steer.SteeredLoss(1.0)(SimpleNamespace(logits=satisfied), labels)
        near_zero.backward()
        self.assertLess(abs(near_zero.item()), 1e-6)
        self.assertLess(satisfied.grad.abs().max().item(), 1e-6)

    def test_shared_schedule_and_samsum_data_defaults(self):
        parsers = (train_sft.parse_args, train_sft_lora.parse_args,
                   train_ewc.parse_args, train_steer.parse_args)
        for parser in parsers:
            with patch.object(sys, "argv", ["train"]):
                args = parser()
            self.assertEqual(args.train_data_path, "data/samsum_emnlp2019/train")
            self.assertEqual(args.dev_data_path, "data/samsum_emnlp2019/dev")
            self.assertEqual(args.num_train_epochs, 2)
            self.assertEqual(args.learning_rate, 1e-5)
            self.assertEqual(args.gradient_accumulation_steps, 64)
            self.assertEqual(args.max_seq_length, 2048)
            self.assertEqual(args.seed, 42)

    def test_report_includes_added_method_and_base_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, score in (("base", 0.4), ("ewc", 0.5)):
                directory = root / "models" / name
                directory.mkdir(parents=True)
                for task in stable_five.TASKS:
                    (directory / f"{task}.json").write_text(
                        '{"metric":"acc,none","score":' + str(score) + ',"effective_samples":1}'
                    )
            args = SimpleNamespace(output_dir=root, models=["base", "ewc"],
                                   model_paths={"base": "base-model", "ewc": "ewc-model"})
            stable_five.write_report(args)
            report = (root / "results.md").read_text()
            self.assertIn("ewc − Base", report)
            self.assertIn("+10.00", report)
            self.assertEqual(len((root / "scores.csv").read_text().splitlines()), 11)

    def test_samsum_extraction(self):
        self.assertEqual(extract_summary("<answer>Short summary</answer>"), "Short summary")


if __name__ == "__main__":
    unittest.main()
