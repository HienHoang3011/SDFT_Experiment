"""Per-sample breakdown of the Stable Five benchmarks: why each model gains or loses points.

Reads <stable-five>/models/<model>/<task>.samples.jsonl written by benchmarks/stable_five/run.py and
re-scores each sample (the samples file stores responses but not per-sample metric values):
  gsm8k      flexible-extract answer vs target; output format and length
  bbh        flexible-extract answer vs target (ignore case); per subtask; unextractable answers
  boolq      log-likelihood choice vs label; predicted-yes rate
  ifeval     lm-eval's own IFEval checker, per instruction type (strict)
  humaneval  executes program + tests in a subprocess (10 s timeout); failure type
Each item is also compared with Base: kept (both right), lost (Base right, model wrong), gained.
"""
import argparse
import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("base", "full_sft", "lora", "steer_lora", "steer_full", "fapm")


def raw(row):
    x = row["raw_response"]
    while isinstance(x, list):
        x = x[0] if x else ""
    return x if isinstance(x, str) else ""


# Generation tasks with two filters store each doc twice, in the order of the task's filter_list.
# Index of the scored "flexible-extract" copy: GSM8K lists strict-match first, BBH flexible-extract first.
SCORED_COPY = {"gsm8k_cot_zeroshot": 1}


def load(directory, model, task):
    path = directory / model / f"{task}.samples.jsonl"
    if not path.exists():
        return None
    copies = defaultdict(list)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        copies[(row["task"], row["doc_id"])].append(row)
    index = SCORED_COPY.get(task, 0)
    return [rows[min(index, len(rows) - 1)] for rows in copies.values()]


def style(text):
    return {"answer_tag": "<answer>" in text, "html_tag": bool(re.search(r"<(ol|ul|li|cell|blockquote|p|output)>", text)),
            "chars": len(text), "repetitive": repetitive(text), "empty": not text.strip()}


def repetitive(text, n=8):
    words = text.split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(grams) >= 20 and Counter(grams).most_common(1)[0][1] >= 4


def number(text):
    found = re.findall(r"-?[\d,]*\.?\d+", text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def gsm8k(rows):
    out = {}
    for r in rows:
        pred = r["filtered_response"][0] if r["filtered_response"] else ""
        gold = r["target"].split("####")[-1].strip().replace(",", "")
        try:
            ok = float(str(pred).replace(",", "").replace("$", "").rstrip(".")) == float(gold)
        except ValueError:
            ok = False
        out[r["doc_id"]] = {"correct": ok, **style(raw(r)), "steps": raw(r).count("=")}
    return out


def bbh(rows):
    out = {}
    for r in rows:
        pred = r["filtered_response"][0] if r["filtered_response"] else ""
        out[(r["task"], r["doc_id"])] = {"correct": pred.strip().lower() == r["target"].strip().lower(),
                                         "invalid": pred == "[invalid]", "subtask": r["task"].removeprefix("bbh_cot_zeroshot_"),
                                         **style(raw(r))}
    return out


def boolq(rows):
    out = {}
    for r in rows:
        scores = [c[0] for c in r["filtered_response"]]
        pred = int(scores[1] > scores[0])
        out[r["doc_id"]] = {"correct": pred == r["target"], "pred_yes": pred, "label": r["target"]}
    return out


def ifeval(rows, docs):
    from lm_eval.tasks.ifeval.utils import process_results
    out = {}
    for r in rows:
        doc = docs[r["doc_id"]]
        res = process_results(doc, [r["filtered_response"][0] if isinstance(r["filtered_response"], list) else r["filtered_response"]])
        out[r["doc_id"]] = {"correct": res["prompt_level_strict_acc"],
                            "per_instruction": dict(zip(doc["instruction_id_list"], res["inst_level_strict_acc"])),
                            **style(raw(r))}
    return out


def run_program(job):
    program, test, entry = job
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "prog.py"
        path.write_text(program + "\n\n" + test + f"\n\ncheck({entry})\n")
        try:
            p = subprocess.run([sys.executable, "-I", str(path)], cwd=tmp, capture_output=True, text=True, timeout=10)
        except subprocess.TimeoutExpired:
            return False, "Timeout"
    if p.returncode == 0:
        return True, None
    last = [l for l in p.stderr.strip().splitlines() if l.strip()]
    return False, (re.match(r"(\w+(Error|Exception)|AssertionError)", last[-1]) or [last[-1][:40] if last else "unknown"])[0]


def humaneval(rows, docs):
    jobs = []
    for r in rows:
        doc = docs[r["doc_id"]]
        program = r["filtered_response"][0] if isinstance(r["filtered_response"][0], str) else r["filtered_response"][0][0]
        jobs.append((program, doc["test"], doc["entry_point"]))
    with mp.Pool(16) as pool:
        results = pool.map(run_program, jobs)
    return {r["doc_id"]: {"correct": ok, "error": err, **style(raw(r))} for r, (ok, err) in zip(rows, results)}


def summarize(items, base):
    keys = list(items)
    n = len(keys)
    s = {"n": n, "acc": sum(items[k]["correct"] for k in keys) / n}
    if base is not None:
        s["lost_vs_base"] = sum(base[k]["correct"] and not items[k]["correct"] for k in keys) / n
        s["gained_vs_base"] = sum(items[k]["correct"] and not base[k]["correct"] for k in keys) / n
    for flag in ("answer_tag", "html_tag", "repetitive", "empty", "invalid"):
        if flag in items[keys[0]]:
            hit = [k for k in keys if items[k][flag]]
            s[f"{flag}_rate"] = len(hit) / n
            s[f"acc_when_{flag}"] = sum(items[k]["correct"] for k in hit) / len(hit) if hit else None
            if base is not None and hit:
                # Of the items lost vs Base, how many show this flag.
                lost = [k for k in keys if base[k]["correct"] and not items[k]["correct"]]
                s[f"lost_with_{flag}"] = sum(items[k][flag] for k in lost) / len(lost) if lost else None
    if "chars" in items[keys[0]]:
        s["median_chars"] = sorted(items[k]["chars"] for k in keys)[n // 2]
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stable-five", type=Path, default=ROOT / "artifacts/samsum/fp32-master-2epoch-20261012/stable-five")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/samsum/fp32-master-2epoch-20261012/benchmark-analysis")
    args = parser.parse_args()
    from datasets import load_dataset
    ife_docs = list(load_dataset("google/IFEval", split="train"))
    he_docs = list(load_dataset("openai/openai_humaneval", split="test"))
    directory = args.stable_five / "models"
    report = {}
    per_item = {}
    for task, fn in [("gsm8k_cot_zeroshot", gsm8k), ("bbh_cot_zeroshot", bbh), ("boolq", boolq),
                     ("ifeval", lambda r: ifeval(r, ife_docs)), ("humaneval_instruct", lambda r: humaneval(r, he_docs))]:
        per_item[task] = {}
        for model in MODELS:
            rows = load(directory, model, task)
            if rows is not None:
                per_item[task][model] = fn(rows)
        base = per_item[task].get("base")
        report[task] = {}
        for model, items in per_item[task].items():
            s = summarize(items, None if model == "base" else base)
            first = next(iter(items.values()))
            if "subtask" in first:
                by = defaultdict(list)
                for v in items.values():
                    by[v["subtask"]].append(v)
                s["subtasks"] = {k: {"acc": sum(x["correct"] for x in v) / len(v),
                                     "invalid": sum(x["invalid"] for x in v) / len(v)} for k, v in sorted(by.items())}
            if "per_instruction" in first:
                by = defaultdict(list)
                for v in items.values():
                    for inst, ok in v["per_instruction"].items():
                        by[inst.split(":")[0]].append(ok)
                s["instruction_groups"] = {k: sum(v) / len(v) for k, v in sorted(by.items())}
            if "error" in first:
                s["errors"] = dict(Counter(v["error"] for v in items.values() if not v["correct"]).most_common())
            if "pred_yes" in first:
                s["pred_yes_rate"] = sum(v["pred_yes"] for v in items.values()) / len(items)
                s["recall_yes"] = sum(v["correct"] for v in items.values() if v["label"] == 1) / max(1, sum(v["label"] for v in items.values()))
                s["recall_no"] = sum(v["correct"] for v in items.values() if v["label"] == 0) / max(1, sum(1 - v["label"] for v in items.values()))
            report[task][model] = s
            print(task, model, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                                           if not isinstance(v, dict)}), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
