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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
sys.path.insert(0, str(ROOT / "eval"))

import train_sft  # noqa: E402
import train_sft_lora  # noqa: E402
import train_steer  # noqa: E402
from eval_samsum import extract_summary  # noqa: E402

spec = importlib.util.spec_from_file_location("stable_five_run", ROOT / "benchmarks/stable_five/run.py")
stable_five = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable_five)


class MethodTests(unittest.TestCase):


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
                   train_steer.parse_args)
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
            for name, score in (("base", 0.4), ("steer_anchor", 0.5)):
                directory = root / "models" / name
                directory.mkdir(parents=True)
                for task in stable_five.TASKS:
                    (directory / f"{task}.json").write_text(
                        '{"metric":"acc,none","score":' + str(score) + ',"effective_samples":1}'
                    )
            args = SimpleNamespace(output_dir=root, models=["base", "steer_anchor"], tasks=list(stable_five.TASKS),
                                   model_paths={"base": "base-model", "steer_anchor": "steer_anchor-model"})
            stable_five.write_report(args)
            report = (root / "results.md").read_text()
            self.assertIn("steer_anchor − Base", report)
            self.assertIn("+10.00", report)
            self.assertEqual(len((root / "scores.csv").read_text().splitlines()), 11)

    def test_samsum_extraction(self):
        self.assertEqual(extract_summary("<answer>Short summary</answer>"), "Short summary")


if __name__ == "__main__":
    unittest.main()
