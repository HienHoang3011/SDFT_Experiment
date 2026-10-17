"""MMLU and HellaSwag with the same runner, environment and ChatML settings as run.py.

These replace BBH (slow chain-of-thought generation) for later experiments. Both are log-likelihood tasks, so they
measure knowledge without being affected by the SAMSum output format; GSM8K, IFEval and HumanEval stay in run.py to
measure format leakage. Arguments are those of run.py; --tasks defaults to the two tasks below.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run  # noqa: E402

KNOWLEDGE_TASKS = {
    "mmlu": {
        "metric": "acc,none",
        "num_fewshot": None,
        "batch_size": 8,
        "gen_kwargs": None,
        "expected_samples": 14042,
    },
    "hellaswag": {
        "metric": "acc_norm,none",
        "num_fewshot": None,
        "batch_size": 8,
        "gen_kwargs": None,
        "expected_samples": 10042,
    },
}
run.TASKS.clear()
run.TASKS.update(KNOWLEDGE_TASKS)
# run.main() starts each worker as `python <run.__file__> --worker ...`; point it here so workers see these tasks.
run.__file__ = str(Path(__file__).resolve())

if __name__ == "__main__":
    run.main()
