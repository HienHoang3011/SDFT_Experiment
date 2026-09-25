from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "legacy_eval"))

from close_choice_tasks.utils import extract_mc_answer, format_mc_prompt  # noqa: E402
from eval_prior_capabilities import (  # noqa: E402
    CANONICAL_TASKS,
    PRIMARY_METRICS,
    STANDARD_GENERATION_TASKS,
    build_standard_generation_samples,
    build_summary,
)


class MultipleChoiceParserTests(unittest.TestCase):
    labels = ["A", "B", "C", "D"]
    choices = ["zero", "four", "two", "six"]

    def parse(self, text: str):
        return extract_mc_answer(text, self.labels, self.choices)

    def assert_label(self, text: str, expected: str) -> None:
        parsed = self.parse(text)
        self.assertTrue(parsed["parse_valid"], parsed)
        self.assertEqual(parsed["predicted_label"], expected)

    def test_requested_and_common_answer_formats(self):
        cases = {
            "Final answer: B": "B",
            "Final answer: (B). Because it is correct.": "B",
            "<answer>B</answer>": "B",
            "<letter>B</letter>": "B",
            "Final answer: <letter>B</letter>": "B",
            "<letter>Final answer: B</letter>": "B",
            "The correct answer is:\n\nB. four": "B",
            "The best answer is: B — four": "B",
            "B. four": "B",
            "The result is four": "B",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assert_label(text, expected)

    def test_choice_text_inside_letter_element(self):
        parsed = extract_mc_answer("<letter>yes</letter>", ["A", "B"], ["no", "yes"])
        self.assertEqual(parsed["predicted_label"], "B")

    def test_last_explicit_answer_wins(self):
        self.assert_label("Answer: A\nFinal answer: B", "B")

    def test_incidental_letters_and_mismatched_labelled_choice_are_rejected(self):
        invalid = [
            "I considered A and B but did not finish.",
            "B. zero",
            "The derivation uses matrix A.",
            "Option Z",
            "",
        ]
        for text in invalid:
            with self.subTest(text=text):
                parsed = self.parse(text)
                self.assertFalse(parsed["parse_valid"], parsed)

    def test_prompt_does_not_invite_literal_xml_placeholder(self):
        prompt = format_mc_prompt("What is 2 + 2?", ["3", "4"])
        self.assertNotIn("<letter>", prompt)
        self.assertNotIn("Final answer: <", prompt)
        self.assertIn("letters: A, B", prompt)
        self.assertNotIn("letters: A, B, C", prompt)
        self.assertIn("Do not include an explanation", prompt)


class StandardGenerationTaskTests(unittest.TestCase):
    def test_standard_math_and_instruct_humaneval_are_separate_tasks(self):
        self.assertEqual(
            STANDARD_GENERATION_TASKS,
            ("minerva_math500", "humaneval_instruct"),
        )
        self.assertNotIn("humaneval", CANONICAL_TASKS)
        self.assertEqual(
            PRIMARY_METRICS["math500"],
            ("results", "minerva_math500", "math_verify,none"),
        )
        self.assertEqual(
            PRIMARY_METRICS["humaneval"],
            ("results", "humaneval_instruct", "pass@1,create_test"),
        )

    def test_summary_uses_math_verify_and_executable_pass_at_one(self):
        canonical = {
            "groups": {"mmlu": {"acc,none": 0.10}},
            "results": {
                "piqa": {"acc_norm,none": 0.20},
                "boolq": {"acc,none": 0.30},
                "ifeval": {
                    "prompt_level_strict_acc,none": 0.40,
                    "prompt_level_loose_acc,none": 0.41,
                    "inst_level_strict_acc,none": 0.42,
                    "inst_level_loose_acc,none": 0.43,
                },
            },
        }
        closed_generation = {
            "groups": {
                "mmlu_gen": {
                    "generation_accuracy,none": 0.11,
                    "parse_valid_rate,none": 0.91,
                }
            },
            "results": {
                "piqa_gen": {
                    "generation_accuracy,none": 0.21,
                    "parse_valid_rate,none": 0.92,
                },
                "boolq_gen": {
                    "generation_accuracy,none": 0.31,
                    "parse_valid_rate,none": 0.93,
                },
            },
        }
        standard = {
            "results": {
                "minerva_math500": {"math_verify,none": 0.80},
                "humaneval_instruct": {"pass@1,create_test": 0.90},
            },
            "n-samples": {
                "minerva_math500": {"original": 500, "effective": 500},
                "humaneval_instruct": {"original": 164, "effective": 164},
            },
        }
        alignment = {
            "mmlu": {"aligned": True},
            "piqa": {"aligned": True},
            "boolq": {"aligned": True},
            "all_aligned": True,
        }

        summary = build_summary(
            canonical, closed_generation, standard, alignment
        )

        self.assertEqual(summary["pure_generation"]["math500"], {
            "math_verify": 0.80,
            "num_samples": 500,
            "lm_eval_task": "minerva_math500",
        })
        self.assertEqual(summary["pure_generation"]["humaneval"], {
            "pass@1": 0.90,
            "num_samples": 164,
            "lm_eval_task": "humaneval_instruct",
        })
        self.assertAlmostEqual(summary["pure_generation"]["average"], 0.70)

    def test_logged_standard_samples_keep_raw_and_filtered_responses(self):
        result = {
            "samples": {
                "minerva_math500": [
                    {
                        "doc_id": 3,
                        "doc": {"problem": "1+1", "answer": "2"},
                        "resps": [["reasoning then 2"]],
                        "filtered_resps": ["2"],
                        "metrics": ["math_verify"],
                    }
                ],
                "humaneval_instruct": [
                    {
                        "doc_id": 4,
                        "doc": {"entry_point": "add"},
                        "resps": [["return a + b"]],
                        "filtered_resps": [["def add(a, b): return a + b"]],
                        "metrics": ["pass@1"],
                    }
                ],
            }
        }
        rows = build_standard_generation_samples(result)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["task"], "math500")
        self.assertEqual(rows[0]["raw_response"], "reasoning then 2")
        self.assertEqual(rows[0]["filtered_response"], "2")
        self.assertEqual(rows[1]["task"], "humaneval")
        self.assertEqual(rows[1]["raw_response"], "return a + b")


if __name__ == "__main__":
    unittest.main()
