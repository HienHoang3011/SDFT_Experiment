"""Why does FAPM work? Compare its mask with other ways of keeping part of the Full SFT update.

Every variant is W_base + D * M (or W_base + a * D) with D = W_SFT - W_base computed in fp32:
  fapm      top-k of t = |D| (1 - lambda / |W_base|) per tensor   (methods/fapm.py)
  magnitude top-k of |D| per tensor                              (same count k as FAPM)
  random    random k elements per tensor
  scaled    a * D with a per tensor so that ||a D|| = ||D * M_fapm||  (same energy, no sparsity)
  alpha<a>  a * D with one fixed a for every tensor, e.g. alpha0.25
  only_late / only_early / only_head / no_head
            the full D on one region and Base elsewhere: late = layers 21-27 + final norm + lm_head,
            early = embeddings + layers 0-20, head = embeddings + final norm + lm_head, no_head = the rest
  layers<a>-<b> / drop<a>-<b>
            the full D only on transformer layers a..b / the full D everywhere except layers a..b
  fapm_special
            fapm, plus the full D on the embedding and lm_head rows of the special tokens (id >= 151643)
  fapm_full<region>
            fapm, but the full D on one region (late / early / head as above), e.g. fapm_fulllate
k = 10% of every tensor, like methods/fapm.py.

  weights  per-tensor statistics of the masks (no model forward)
  probe    builds each variant in GPU memory and measures
           SAMSum test ROUGE-L (same prompt / decoding as eval/eval_samsum.py),
           GSM8K test accuracy on a fixed subset (greedy, chat template, flexible last-number match),
           BoolQ validation accuracy (ChatML, no/yes logit like eval/analyze_boolq_margin.py),
           and KL(Base || model) on SAMSum dev and GSM8K gold answers (teacher forced).
"""
import argparse
import gc
import json
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
sys.path.insert(0, str(ROOT / "eval"))

VARIANTS = ("base", "full_sft", "fapm", "magnitude", "random", "scaled", "only_late", "only_early", "only_head", "no_head",
            "fapm_special", "fapm_fulllate", "fapm_fullearly", "fapm_fullhead")
LATE_FROM = 21
SPECIAL_FROM = 151643  # Qwen2 special tokens (<|endoftext|>, <|im_start|>, <|im_end|>) start here


def region(key):
    if key.startswith("model.layers."):
        return "late" if int(key.split(".")[2]) >= LATE_FROM else "early"
    return "head"


def keeps(variant, key):
    span = re.fullmatch(r"(layers|drop)(\d+)-(\d+)", variant)
    if span:
        inside = key.startswith("model.layers.") and int(span[2]) <= int(key.split(".")[2]) <= int(span[3])
        return inside if span[1] == "layers" else not inside
    r = region(key)
    return {"only_late": r == "late" or key in ("model.norm.weight", "lm_head.weight"),
            "only_early": r == "early" or key == "model.embed_tokens.weight",
            "only_head": r == "head", "no_head": r != "head"}[variant]


class Store:
    def __init__(self, path):
        self.index = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {f: safe_open(str(Path(path) / f), "pt") for f in set(self.index.values())}

    def get(self, key):
        return self.handles[self.index[key]].get_tensor(key)


def top_mask(score, k):
    mask = torch.zeros(score.numel(), dtype=torch.bool, device=score.device)
    mask[torch.topk(score.reshape(-1), k).indices] = True
    return mask.view_as(score)


def masks(w, d, k, seed):
    """FAPM / magnitude / random masks for one tensor, all with exactly k elements."""
    lam = w.abs().mean()
    t = d.abs() - lam * (d.abs() / (w.abs() + 1e-8))  # identical to methods/fapm.py
    g = torch.Generator(device=w.device).manual_seed(seed)
    return {"fapm": top_mask(t, k), "magnitude": top_mask(d.abs(), k),
            "random": top_mask(torch.rand(w.shape, generator=g, device=w.device), k)}, t, lam


def module_of(key):
    return key.split(".")[-2] if key.startswith("model.layers.") else key.removesuffix(".weight")


def weight_stats(base, sft, ratio, out):
    rows = []
    for i, key in enumerate(sorted(base.index)):
        w = base.get(key).cuda().float()
        d = sft.get(key).cuda().float() - w
        k = int(d.numel() * ratio)
        if k == 0:
            continue
        m, t, lam = masks(w, d, k, i)
        e = d.square()
        row = {"tensor": key, "module": module_of(key), "numel": d.numel(), "k": k,
               "changed": int(d.ne(0).sum()), "positive_t": int(t.gt(0).sum()),
               "w_sq": w.square().sum().item(), "delta_sq": e.sum().item(), "lambda": lam.item(),
               "fapm_mag_overlap": int((m["fapm"] & m["magnitude"]).sum())}
        for name, mask in m.items():
            kept = d[mask]
            row[f"{name}_energy"] = e[mask].sum().item()
            row[f"{name}_mean_abs_w"] = w.abs()[mask].mean().item()
            row[f"{name}_mean_abs_d"] = kept.abs().mean().item()
            # Relative perturbation of the kept elements: sum D^2 / sum W^2 on those elements.
            row[f"{name}_rel_sq"] = (e[mask].sum() / w[mask].square().sum().clamp_min(1e-30)).item()
            # Fraction of kept updates that shrink |w| (D and W of opposite sign).
            row[f"{name}_shrink"] = (kept.sign() * w[mask].sign()).lt(0).float().mean().item()
        row["all_mean_abs_w"] = w.abs().mean().item()
        row["all_mean_abs_d"] = d.abs().mean().item()
        row["all_rel_sq"] = row["delta_sq"] / row["w_sq"]
        row["all_shrink"] = (d.sign() * w.sign()).lt(0).float().mean().item()
        rows.append(row)
        del w, d, m, t, e
        if i % 50 == 0:
            print(f"weights {i + 1}/{len(base.index)}", flush=True)
    torch.cuda.empty_cache()

    def total(group):
        n = sum(r["numel"] for r in group)
        kk = sum(r["k"] for r in group)
        de = sum(r["delta_sq"] for r in group)
        res = {"numel": n, "k": kk, "changed_frac": sum(r["changed"] for r in group) / n,
               "positive_t_frac": sum(r["positive_t"] for r in group) / n,
               "fapm_magnitude_overlap": sum(r["fapm_mag_overlap"] for r in group) / kk,
               "full_rel_norm": (de / sum(r["w_sq"] for r in group)) ** 0.5,
               "all_shrink": sum(r["all_shrink"] * r["numel"] for r in group) / n}
        for name in ("fapm", "magnitude", "random"):
            res[f"{name}_energy_kept"] = sum(r[f"{name}_energy"] for r in group) / de
            res[f"{name}_rel_norm"] = (sum(r[f"{name}_energy"] for r in group) / sum(r["w_sq"] for r in group)) ** 0.5
            res[f"{name}_mean_abs_w"] = sum(r[f"{name}_mean_abs_w"] * r["k"] for r in group) / kk
            res[f"{name}_mean_abs_d"] = sum(r[f"{name}_mean_abs_d"] * r["k"] for r in group) / kk
            res[f"{name}_shrink"] = sum(r[f"{name}_shrink"] * r["k"] for r in group) / kk
            res[f"{name}_kept_rel_norm"] = (sum(r[f"{name}_rel_sq"] * r["k"] for r in group) / kk) ** 0.5
        res["all_mean_abs_w"] = sum(r["all_mean_abs_w"] * r["numel"] for r in group) / n
        res["all_mean_abs_d"] = sum(r["all_mean_abs_d"] * r["numel"] for r in group) / n
        return res

    modules = sorted({r["module"] for r in rows})
    summary = {"k_ratio": ratio, "whole_model": total(rows),
               "by_module": {m: total([r for r in rows if r["module"] == m]) for m in modules},
               "by_layer": {str(l): total([r for r in rows if r["tensor"].startswith(f"model.layers.{l}.")]) for l in range(28)}}
    (out / "weights_per_tensor.json").write_text(json.dumps(rows) + "\n")
    (out / "weights_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["whole_model"], indent=1), flush=True)
    return summary


@torch.no_grad()
def build(model, base_cpu, sft, variant, ratio):
    for i, (key, param) in enumerate(model.named_parameters()):
        w = base_cpu[key].cuda().float()
        if variant == "base":
            param.copy_(w.to(param.dtype))
            continue
        d = sft.get(key).cuda().float() - w
        k = int(d.numel() * ratio)
        if variant == "full_sft" or k == 0:
            new = w + d if variant == "full_sft" else w
        elif variant in ("only_late", "only_early", "only_head", "no_head") or re.fullmatch(r"(layers|drop)\d+-\d+", variant):
            new = w + d if keeps(variant, key) else w
        elif variant.startswith("alpha"):
            new = w + float(variant[5:]) * d
        elif variant == "scaled":
            m, _, _ = masks(w, d, k, i)
            total = d.square().sum()
            new = w + d * (d[m["fapm"]].square().sum() / total).sqrt() if total > 0 else w
        elif variant.startswith("fapm_full"):
            m, _, _ = masks(w, d, k, i)
            new = w + d if keeps("only_" + variant.removeprefix("fapm_full"), key) else w + d * m["fapm"]
        elif variant == "fapm_special":
            m, _, _ = masks(w, d, k, i)
            keep = m["fapm"].float()
            if key in ("model.embed_tokens.weight", "lm_head.weight"):
                keep[SPECIAL_FROM:] = 1
            new = w + d * keep
        else:
            m, _, _ = masks(w, d, k, i)
            new = w + d * m[variant]
        param.copy_(new.to(param.dtype))
        del w, d, new
    torch.cuda.empty_cache()


@torch.inference_mode()
def generate(model, tokenizer, prompts, max_new_tokens, batch_size, prefill="", stats=None):
    """stats, if a list, receives per output: whether it stopped with the end-of-turn token, and the raw text
    with special tokens kept (to spot <|im_end|> spelled out as ordinary tokens)."""
    outputs = []
    for start in range(0, len(prompts), batch_size):
        texts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) + prefill
                 for p in prompts[start:start + batch_size]]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        new = gen[:, enc.input_ids.shape[1]:]
        outputs += tokenizer.batch_decode(new, skip_special_tokens=True)
        if stats is not None:
            stats += [{"stopped": bool((row == tokenizer.eos_token_id).any()), "text": text}
                      for row, text in zip(new, tokenizer.batch_decode(new, skip_special_tokens=False))]
    return outputs


def output_style(answers, stats):
    n = len(answers)
    return {"stop_rate": sum(s["stopped"] for s in stats) / n,
            "spelled_end_rate": sum("<|im_end|>" in a or "im_end" in a for a in answers) / n,
            "answer_tag_rate": sum("<answer>" in a for a in answers) / n,
            "median_chars": sorted(len(a) for a in answers)[n // 2]}


def last_number(text):
    found = re.findall(r"-?[\d,]*\.?\d+", text)
    return found[-1].replace(",", "").rstrip(".") if found else None


def same_number(a, b):
    try:
        return a is not None and abs(float(a) - float(b)) < 1e-6
    except ValueError:
        return False


@torch.inference_mode()
def teacher_forced(model, probes):
    """Full-vocabulary log-probs at assistant positions, fp32, kept on CPU."""
    for ids, positions in probes:
        logits = model(input_ids=ids.cuda(), use_cache=False).logits[0, positions].float()
        yield F.log_softmax(logits, -1).cpu()


def chat_probes(tokenizer, conversations, max_tokens=64):
    probes = []
    for messages in conversations:
        enc = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        ids, mask = enc["input_ids"], enc["assistant_masks"]
        positions = [i - 1 for i in range(1, len(ids)) if mask[i]]
        if len(positions) > max_tokens:
            positions = [positions[j] for j in torch.linspace(0, len(positions) - 1, max_tokens).round().long().tolist()]
        probes.append((torch.tensor([ids]), positions))
    return probes


def drift(reference, current, probes):
    kl, nll, n = 0.0, 0.0, 0
    for (ids, positions), p, q in zip(probes, reference, current):
        targets = ids[0, torch.tensor(positions) + 1]
        kl += (p.exp() * (p - q)).sum().item()
        nll += -q.gather(1, targets[:, None]).sum().item()
        n += len(positions)
    return kl / n, nll / n


@torch.inference_mode()
def boolq(model, tokenizer, docs, choice_ids, batch_size=8):
    correct, yes = 0, 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start:start + batch_size]
        prompts = [tokenizer.apply_chat_template([{"role": "user", "content": f"{r['passage']}\nQuestion: {r['question']}?\nAnswer:"}],
                                                 tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        pred = model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1][:, choice_ids].argmax(-1).cpu()
        labels = torch.tensor([r["label"] for r in batch])
        correct += int((pred == labels).sum())
        yes += int(pred.sum())
    return correct / len(docs), yes / len(docs)


def probe(args, out):
    from datasets import Dataset, load_dataset
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen2_base import configure_qwen2_base_tokenizer, sync_model_special_tokens
    from eval_samsum import extract_summary

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, padding_side="left")
    configure_qwen2_base_tokenizer(tokenizer, args.base_model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    sync_model_special_tokens(model, tokenizer)
    base_cpu = {k: v.detach().cpu().clone() for k, v in model.named_parameters()}
    sft = Store(args.sft_model)

    samsum_test = Dataset.load_from_disk(str(ROOT / "data/samsum_emnlp2019/test"))
    samsum_dev = Dataset.load_from_disk(str(ROOT / "data/samsum_emnlp2019/dev"))
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    gsm_idx = torch.randperm(len(gsm), generator=torch.Generator().manual_seed(0))[:args.gsm8k_n].tolist()
    gsm_rows = [gsm[i] for i in gsm_idx]
    docs = [json.loads(l) for l in Path(args.boolq).read_text().splitlines()][:args.boolq_n]
    choice_ids = torch.tensor([tokenizer.encode(a, add_special_tokens=False)[0] for a in ("no", "yes")])
    dev_probes = chat_probes(tokenizer, [samsum_dev[i]["messages"] for i in range(args.dev_n)])
    gsm_probes = chat_probes(tokenizer, [[{"role": "user", "content": f"Q: {r['question']}\nA: Let's think step by step."},
                                          {"role": "assistant", "content": r["answer"]}] for r in gsm_rows])
    ife_prompts = [r["prompt"] for r in load_dataset("google/IFEval", split="train")][:args.ifeval_n]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    reference = None
    results_path = out / "probe_results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}
    for variant in args.variants:
        if variant in results and variant != "base":
            continue
        started = time.monotonic()
        build(model, base_cpu, sft, variant, args.k_ratio)
        if variant == "base":
            reference = (list(teacher_forced(model, dev_probes)), list(teacher_forced(model, gsm_probes)))
            if "base" in results:
                continue
        row = {}
        row["samsum_dev_kl"], row["samsum_dev_nll"] = drift(reference[0], teacher_forced(model, dev_probes), dev_probes)
        row["gsm8k_gold_kl"], row["gsm8k_gold_nll"] = drift(reference[1], teacher_forced(model, gsm_probes), gsm_probes)
        row["boolq_acc"], row["boolq_yes_rate"] = boolq(model, tokenizer, docs, choice_ids)
        stats = []
        answers = generate(model, tokenizer, [f"Q: {r['question']}\nA: Let's think step by step." for r in gsm_rows], 512, args.batch_size,
                           args.gsm8k_prefill, stats)
        row.update({f"gsm8k_{k}": v for k, v in output_style(answers, stats).items()})
        row["gsm8k_acc"] = sum(same_number(last_number(a), r["answer"].split("####")[-1].strip()) for a, r in zip(answers, gsm_rows)) / len(gsm_rows)
        n = len(samsum_test) if args.samsum_n <= 0 else args.samsum_n
        summaries = generate(model, tokenizer, samsum_test["prompt"][:n], 128, args.batch_size)
        row["samsum_rougeL"] = sum(scorer.score(ref, extract_summary(s))["rougeL"].fmeasure
                                   for ref, s in zip(samsum_test["summary"][:n], summaries)) / n
        extra = {}
        if args.ifeval_n > 0:
            stats = []
            extra["ifeval"] = generate(model, tokenizer, ife_prompts, 512, args.batch_size, stats=stats)
            row.update({f"ifeval_{k}": v for k, v in output_style(extra["ifeval"], stats).items()})
            row["ifeval_empty_rate"] = sum(not a.split("<|im_end|>")[0].strip() for a in extra["ifeval"]) / len(ife_prompts)
        row["seconds"] = time.monotonic() - started
        results[variant] = row
        results_path.write_text(json.dumps(results, indent=2) + "\n")
        (out / f"samples_{variant}.json").write_text(json.dumps({"gsm8k": answers[:20], "samsum": summaries[:20], **{k: v[:20] for k, v in extra.items()}}, ensure_ascii=False, indent=1) + "\n")
        print(variant, json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["weights", "probe"])
    parser.add_argument("--base-model", default=str(ROOT / "artifacts/base/Qwen2-7B"))
    parser.add_argument("--sft-model", required=True, help="Full SFT checkpoint (fp32 weights)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--k-ratio", type=float, default=0.1)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), help=f"{', '.join(VARIANTS)} or alpha<a>")
    parser.add_argument("--gsm8k-prefill", default="", help="text placed at the start of the assistant turn for GSM8K")
    parser.add_argument("--boolq", default=str(ROOT / "artifacts/samsum/fair-methods-2epoch-log25-tmux/reports/margin-mechanism/boolq.validation.jsonl"))
    parser.add_argument("--boolq-n", type=int, default=1000)
    parser.add_argument("--gsm8k-n", type=int, default=250)
    parser.add_argument("--samsum-n", type=int, default=0, help="0 = all 819 test dialogues")
    parser.add_argument("--dev-n", type=int, default=200)
    parser.add_argument("--ifeval-n", type=int, default=0, help="also generate on the first N IFEval prompts (stop / empty rates)")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if args.variants[0] != "base" and args.mode == "probe":
        args.variants = ["base"] + [v for v in args.variants if v != "base"]
    bad = [v for v in args.variants if v not in VARIANTS and not re.fullmatch(r"alpha\d*\.?\d+|(layers|drop)\d+-\d+", v)]
    if bad:
        parser.error(f"unknown variants: {bad}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "weights":
        weight_stats(Store(args.base_model), Store(args.sft_model), args.k_ratio, args.output_dir)
    else:
        probe(args, args.output_dir)


if __name__ == "__main__":
    main()
