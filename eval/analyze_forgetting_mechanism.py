"""Mechanism of forgetting for the fp32 rerun: weight updates, logits, gradients.

  spectrum   SVD of the update D = W_model - W_base on sampled matrices (every 3rd layer, all 7 projections):
             relative norm, stable rank, energy in the top singular directions of D, and the share of D's
             energy inside W_base's top-1% singular subspace (left side) compared with a random direction.
             LoRA is merged as (alpha / r) B A.
  logits     On ChatML prompts of GSM8K, BBH, IFEval and SAMSum test, teacher-force Base's own lm-eval answer
             and compare every model with Base per position: KL(Base || model) at the first answer token vs
             later tokens, probability that the answer opens with a "<" tag token, probability of an
             immediate end of turn, and a logit lens of those two probabilities over layers at the first token.
  gradients  CE gradient at Base on SAMSum train (task) and on Base's own GSM8K / IFEval answers (general),
             cosine per module, and the first-order loss change g . D for every model's update compared with
             the measured loss change.
"""
import argparse
import gc
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "methods"))
RUN = ROOT / "artifacts/samsum/fp32-master-2epoch-20261012"
OLD = ROOT / "artifacts/samsum/fair-methods-2epoch-log25-tmux/models"
MODELS = {"full_sft": "/root/sdft-fp32-runs/models/full_sft", "steer_full": "/root/sdft-fp32-runs/models/steer_full",
          "fapm": "/root/sdft-fp32-runs/models/fapm", "lora": str(OLD / "lora"), "steer_lora": str(OLD / "steer_lora")}
BASE = ROOT / "artifacts/base/Qwen2-7B"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class Store:
    def __init__(self, path):
        self.index = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
        self.handles = {f: safe_open(str(Path(path) / f), "pt") for f in set(self.index.values())}

    def get(self, key):
        return self.handles[self.index[key]].get_tensor(key)


class Lora:
    def __init__(self, path):
        config = json.loads((Path(path) / "adapter_config.json").read_text())
        self.scale = config["lora_alpha"] / config["r"]
        self.handle = safe_open(str(Path(path) / "adapter_model.safetensors"), "pt")
        self.keys = set(self.handle.keys())

    def delta(self, key):
        stem = "base_model.model." + key.removesuffix(".weight")
        a, b = f"{stem}.lora_A.weight", f"{stem}.lora_B.weight"
        if a not in self.keys:
            return None
        return self.scale * (self.handle.get_tensor(b).cuda().float() @ self.handle.get_tensor(a).cuda().float())


def deltas(key, w, stores):
    out = {}
    for name, store in stores.items():
        if isinstance(store, Lora):
            d = store.delta(key)
            out[name] = d if d is not None else torch.zeros_like(w)
        else:
            out[name] = store.get(key).cuda().float() - w
    return out


def open_stores():
    return {n: (Lora(p) if (Path(p) / "adapter_config.json").exists() else Store(p)) for n, p in MODELS.items()}


# ----------------------------------------------------------------------------------------------- spectrum
def spectrum(out):
    base, stores = Store(BASE), open_stores()
    rows = []
    for layer in range(0, 28, 3):
        for proj in PROJ:
            part = "self_attn" if proj in PROJ[:4] else "mlp"
            key = f"model.layers.{layer}.{part}.{proj}.weight"
            w = base.get(key).cuda().float()
            u, s, _ = torch.linalg.svd(w, full_matrices=False)
            r = max(1, int(0.01 * s.numel()))
            top = u[:, :r]
            for name, d in deltas(key, w, stores).items():
                if d.abs().sum() == 0:
                    continue
                sd = torch.linalg.svdvals(d)
                energy = sd.square()
                total = energy.sum()
                rows.append({"layer": layer, "proj": proj, "model": name,
                             "rel_norm": (total.sqrt() / w.norm()).item(),
                             "stable_rank": (total / energy[0]).item(),
                             "top1_energy": (energy[0] / total).item(),
                             "top16_energy": (energy[:16].sum() / total).item(),
                             "top256_energy": (energy[:256].sum() / total).item(),
                             "in_w_top1pct": ((top.T @ d).square().sum() / total).item(),
                             "random_expect": r / w.shape[0]})
                del d, sd
            print("spectrum", key, flush=True)
            del w, u, s
            torch.cuda.empty_cache()
    (out / "spectrum_rows.json").write_text(json.dumps(rows) + "\n")
    summary = defaultdict(dict)
    for name in MODELS:
        for proj in PROJ + ("all",):
            sel = [r for r in rows if r["model"] == name and (proj == "all" or r["proj"] == proj)]
            if sel:
                summary[name][proj] = {k: sum(r[k] for r in sel) / len(sel) for k in
                                       ("rel_norm", "stable_rank", "top1_energy", "top16_energy", "top256_energy",
                                        "in_w_top1pct", "random_expect")}
    (out / "spectrum_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name in summary:
        print(name, json.dumps({k: round(v, 4) for k, v in summary[name]["all"].items()}))


# ----------------------------------------------------------------------------------------------- logits
def raw(row):
    x = row["raw_response"]
    while isinstance(x, list):
        x = x[0] if x else ""
    return x if isinstance(x, str) else ""


def probes(n, seed=0):
    """(task, user prompt, Base's own lm-eval answer) triples."""
    from datasets import Dataset, load_dataset
    rng = random.Random(seed)
    samples = RUN / "stable-five/models/base"
    out = []

    def answers(task):
        found = {}
        for line in (samples / f"{task}.samples.jsonl").read_text().splitlines():
            r = json.loads(line)
            found.setdefault((r["task"], r["doc_id"]), raw(r))
        return found

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    a = answers("gsm8k_cot_zeroshot")
    for i in rng.sample(range(len(gsm)), n):
        out.append(("gsm8k", f"Q: {gsm[i]['question']}\nA: Let's think step by step.", a[("gsm8k_cot_zeroshot", i)]))
    ife = load_dataset("google/IFEval", split="train")
    a = answers("ifeval")
    for i in rng.sample(range(len(ife)), n):
        out.append(("ifeval", ife[i]["prompt"], a[("ifeval", i)]))
    a = answers("bbh_cot_zeroshot")
    keys = rng.sample(sorted(a), n)
    for task, i in keys:
        doc = load_dataset("SaylorTwift/bbh", task.removeprefix("bbh_cot_zeroshot_"), split="test")[i]
        out.append(("bbh", f"Q: {doc['input']}\nA: Let's think step by step.", a[(task, i)]))
    test = Dataset.load_from_disk(str(ROOT / "data/samsum_emnlp2019/test"))
    for i in rng.sample(range(len(test)), n):
        out.append(("samsum", test[i]["prompt"], test[i]["summary"]))
    return out


def load_model(name, path, tokenizer):
    from transformers import AutoModelForCausalLM
    from qwen2_base import sync_model_special_tokens
    adapter = (Path(path) / "adapter_config.json").exists()
    model = AutoModelForCausalLM.from_pretrained(str(BASE) if adapter else path, dtype=torch.bfloat16, device_map="cuda")
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, path).merge_and_unload()
    sync_model_special_tokens(model, tokenizer)
    return model.eval()


@torch.inference_mode()
def run_model(model, encoded, tag_ids, end_ids, max_tokens):
    """Per probe: log-probs at answer positions (fp32, CPU) and logit-lens tag/end probabilities at the first token."""
    inner = model.model
    for ids, start in encoded:
        out = model(input_ids=ids.cuda(), output_hidden_states=True, use_cache=False)
        positions = list(range(start - 1, min(ids.shape[1] - 1, start - 1 + max_tokens)))
        logp = F.log_softmax(out.logits[0, positions].float(), -1).cpu()
        lens = []
        for h in out.hidden_states[1:]:
            p = F.softmax(model.lm_head(inner.norm(h[0, start - 1])).float(), -1)
            lens.append((p[tag_ids].sum().item(), p[end_ids].sum().item()))
        yield logp, lens
        del out


def logits(out, n, max_tokens):
    from transformers import AutoTokenizer
    from qwen2_base import configure_qwen2_base_tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    configure_qwen2_base_tokenizer(tokenizer, str(BASE))
    vocab = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    special = set(tokenizer.all_special_ids)
    tag_ids = torch.tensor([i for i, t in enumerate(vocab) if t and t.startswith("<") and i not in special])
    end_ids = torch.tensor(tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"]))
    items = probes(n)
    encoded = []
    for _, prompt, answer in items:
        head = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
        start = len(tokenizer(head, add_special_tokens=False).input_ids)
        ids = tokenizer(head + answer + "<|im_end|>", add_special_tokens=False, return_tensors="pt").input_ids[:, :start + max_tokens + 1]
        encoded.append((ids, start))
    names = ["base"] + list(MODELS)
    path = out / "logits_summary.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    reference = None
    for name in names:
        if name in results and name != "base":
            continue  # resume; Base is always recomputed as the reference
        model = load_model(name, str(BASE) if name == "base" else MODELS[name], tokenizer)
        current = list(run_model(model, encoded, tag_ids.cuda(), end_ids.cuda(), max_tokens))
        del model
        gc.collect()  # HF modules hold reference cycles; without this the old model stays on the GPU
        torch.cuda.empty_cache()
        if reference is None:
            reference = current
        acc = defaultdict(lambda: defaultdict(list))
        for (task, _, _), (p, _), (q, lens) in zip(items, reference, current):
            kl = (p.exp() * (p - q)).sum(-1)
            acc[task]["kl_first"].append(kl[0].item())
            acc[task]["kl_rest"].append(kl[1:].mean().item() if len(kl) > 1 else 0.0)
            pq = q[0].exp()
            acc[task]["p_tag_first"].append(pq[tag_ids].sum().item())
            acc[task]["p_end_first"].append(pq[end_ids].sum().item())
            acc[task]["lens_tag"].append([t for t, _ in lens])
            acc[task]["lens_end"].append([e for _, e in lens])
        results[name] = {}
        for task, v in acc.items():
            results[name][task] = {k: sum(x) / len(x) for k, x in v.items() if not k.startswith("lens")}
            for k in ("lens_tag", "lens_end"):
                results[name][task][k] = [sum(col) / len(col) for col in zip(*v[k])]
        print(name, json.dumps({t: {k: round(x, 4) for k, x in r.items() if not k.startswith("lens")} for t, r in results[name].items()}), flush=True)
        path.write_text(json.dumps(results, indent=2) + "\n")


# ----------------------------------------------------------------------------------------------- gradients
def gradients(out, n):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from qwen2_base import configure_qwen2_base_tokenizer
    from datasets import Dataset
    tokenizer = AutoTokenizer.from_pretrained(str(BASE))
    configure_qwen2_base_tokenizer(tokenizer, str(BASE))
    rng = random.Random(1)
    train = Dataset.load_from_disk(str(ROOT / "data/samsum_emnlp2019/train"))
    sets = {"samsum_train": [train[i]["messages"] for i in rng.sample(range(len(train)), n)]}
    general = [(p, a) for t, p, a in probes(n, seed=1) if t in ("gsm8k", "ifeval")]
    sets["general"] = [[{"role": "user", "content": p}, {"role": "assistant", "content": a}] for p, a in general]

    def batch(messages):
        enc = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True)
        ids = torch.tensor([enc["input_ids"][:1536]])
        labels = ids.clone()
        labels[~torch.tensor([enc["assistant_masks"][:1536]], dtype=torch.bool)] = -100
        return ids, labels

    model = AutoModelForCausalLM.from_pretrained(str(BASE), torch_dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    grads, losses = {}, {}
    for name, conversations in sets.items():
        model.zero_grad(set_to_none=True)
        data = [batch(c) for c in conversations]
        total = sum(int(l[:, 1:].ne(-100).sum()) for _, l in data)
        loss_sum = 0.0
        for ids, labels in data:
            logits = model(input_ids=ids.cuda()).logits[0, :-1].float()
            loss = F.cross_entropy(logits, labels[0, 1:].cuda(), ignore_index=-100, reduction="sum") / total
            loss.backward()
            loss_sum += loss.item()
        losses[name] = loss_sum
        grads[name] = {k: p.grad.float().cpu() for k, p in model.named_parameters()}
        print("gradient", name, "tokens", total, "loss", loss_sum, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    base, stores = Store(BASE), open_stores()
    acc = defaultdict(lambda: defaultdict(float))
    for i, key in enumerate(sorted(base.index)):
        w = base.get(key).cuda().float()
        gt, gg = grads["samsum_train"][key].cuda(), grads["general"][key].cuda()
        module = key.split(".")[-2] if key.startswith("model.layers.") else key.removesuffix(".weight")
        for group in ("all", module):
            a = acc[group]
            a["gt_gg"] += (gt * gg).sum().item()
            a["gt_sq"] += gt.square().sum().item()
            a["gg_sq"] += gg.square().sum().item()
        for name, d in deltas(key, w, stores).items():
            for group in ("all", module):
                a = acc[group]
                a[f"{name}_gt_d"] += (gt * d).sum().item()
                a[f"{name}_gg_d"] += (gg * d).sum().item()
                a[f"{name}_d_sq"] += d.square().sum().item()
        if i % 50 == 0:
            print("dot", i, flush=True)
    summary = {"losses_at_base": losses, "groups": {}}
    for group, a in acc.items():
        row = {"cos_task_general": a["gt_gg"] / (a["gt_sq"] * a["gg_sq"]) ** 0.5}
        for name in MODELS:
            dn = a[f"{name}_d_sq"] ** 0.5
            if dn == 0:
                continue
            row[name] = {"first_order_task_dloss": a[f"{name}_gt_d"], "first_order_general_dloss": a[f"{name}_gg_d"],
                         "cos_delta_neg_task_grad": -a[f"{name}_gt_d"] / (dn * a["gt_sq"] ** 0.5),
                         "cos_delta_neg_general_grad": -a[f"{name}_gg_d"] / (dn * a["gg_sq"] ** 0.5)}
        summary["groups"][group] = row
    (out / "gradients_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["groups"]["all"], indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["spectrum", "logits", "gradients"])
    parser.add_argument("--output-dir", type=Path, default=RUN / "forgetting-mechanism")
    parser.add_argument("--n", type=int, default=100, help="probes per task")
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    {"spectrum": lambda: spectrum(args.output_dir), "logits": lambda: logits(args.output_dir, args.n, args.max_tokens),
     "gradients": lambda: gradients(args.output_dir, args.n)}[args.mode]()


if __name__ == "__main__":
    main()
