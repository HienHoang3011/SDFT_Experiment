import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from qwen2_base import (
    QWEN2_BASE_MODEL_ID,
    configure_qwen2_base_tokenizer,
    sync_model_special_tokens,
)


class WeightedMetricAccumulator:
    """Accumulate weighted scalar metrics without retaining autograd graphs."""

    def __init__(self):
        self._metrics = {"train": {}, "eval": {}}
        self._max_metrics = {"train": {}, "eval": {}}

    def add(self, mode, name, numerator, denominator):
        numerator = numerator.detach().float()
        denominator = torch.as_tensor(
            denominator,
            device=numerator.device,
            dtype=torch.float32,
        ).detach()
        bucket = self._metrics[mode]
        if name in bucket:
            old_numerator, old_denominator = bucket[name]
            bucket[name] = (
                old_numerator + numerator,
                old_denominator + denominator,
            )
        else:
            bucket[name] = (numerator, denominator)

    def add_max(self, mode, name, value):
        value = value.detach().float()
        bucket = self._max_metrics[mode]
        if name in bucket:
            bucket[name] = torch.maximum(bucket[name], value)
        else:
            bucket[name] = value

    def pop(self, mode):
        metrics = self._metrics[mode]
        max_metrics = self._max_metrics[mode]
        self._metrics[mode] = {}
        self._max_metrics[mode] = {}
        return metrics, max_metrics


class SteeredLoss:
    """Trainer-compatible steered loss plus pre/post-margin diagnostics."""

    def __init__(self, x_factor, metric_accumulator=None):
        if x_factor <= -1.0:
            raise ValueError("x_factor must be greater than -1.0")
        self.margin = math.log1p(x_factor)
        self.metric_accumulator = metric_accumulator
        self.model = None

    def bind_model(self, model):
        self.model = model

    def _record_metrics(
        self,
        active_logits,
        target_logits,
        current_gt_logits,
        target_gt_logits,
        max_other_logits,
    ):
        if self.metric_accumulator is None:
            return

        mode = "train" if self.model is None or self.model.training else "eval"
        num_tokens = active_logits.new_tensor(active_logits.size(0))
        num_logits = active_logits.new_tensor(active_logits.numel())
        activated = current_gt_logits.lt(max_other_logits + self.margin)
        gt_logit_boost = target_gt_logits - current_gt_logits

        self.metric_accumulator.add(
            mode, "logit_mean_before", active_logits.sum(), num_logits
        )
        self.metric_accumulator.add(
            mode, "logit_mean_after", target_logits.sum(), num_logits
        )
        self.metric_accumulator.add(
            mode,
            "top1_logit_before",
            active_logits.max(dim=-1).values.sum(),
            num_tokens,
        )
        self.metric_accumulator.add(
            mode,
            "top1_logit_after",
            target_logits.max(dim=-1).values.sum(),
            num_tokens,
        )
        self.metric_accumulator.add(
            mode,
            "bottom1_logit_before",
            active_logits.min(dim=-1).values.sum(),
            num_tokens,
        )
        self.metric_accumulator.add(
            mode,
            "bottom1_logit_after",
            target_logits.min(dim=-1).values.sum(),
            num_tokens,
        )
        self.metric_accumulator.add(
            mode, "gt_logit_before", current_gt_logits.sum(), num_tokens
        )
        self.metric_accumulator.add(
            mode, "gt_logit_after", target_gt_logits.sum(), num_tokens
        )
        self.metric_accumulator.add(
            mode,
            "margin_gap_before",
            (current_gt_logits - max_other_logits).sum(),
            num_tokens,
        )
        self.metric_accumulator.add(
            mode,
            "margin_gap_after",
            (target_gt_logits - max_other_logits).sum(),
            num_tokens,
        )
        self.metric_accumulator.add(
            mode, "activated_fraction", activated.sum(), num_tokens
        )
        self.metric_accumulator.add(
            mode,
            "gt_logit_boost_activated_mean",
            gt_logit_boost.sum(),
            activated.sum(),
        )
        self.metric_accumulator.add_max(
            mode,
            "gt_logit_boost_activated_max",
            gt_logit_boost.max(),
        )

    def __call__(self, outputs, labels, num_items_in_batch=None):
        if labels is None:
            raise ValueError("Steered loss requires labels")

        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]

        flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        flat_labels = shift_labels.reshape(-1)
        active_mask = flat_labels.ne(-100)

        # A fully masked micro-batch can occur after truncation. Return a zero
        # that stays connected to the graph so backward/DDP remain valid.
        if not active_mask.any():
            return logits.sum() * 0.0

        # Compute the KL in fp32: this loss can be very small once the margin is
        # satisfied, so bf16 precision is not sufficient here.
        active_logits = flat_logits[active_mask].float()
        active_labels = flat_labels[active_mask]

        with torch.no_grad():
            target_logits = active_logits.detach().clone()
            other_logits = target_logits.clone()
            row_indices = torch.arange(target_logits.size(0), device=target_logits.device)
            other_logits[row_indices, active_labels] = float("-inf")

            max_other_logits = other_logits.max(dim=-1).values
            current_gt_logits = target_logits[row_indices, active_labels]
            target_gt_logits = torch.maximum(
                current_gt_logits,
                max_other_logits + self.margin,
            )
            target_logits[row_indices, active_labels] = target_gt_logits
            target_probs = F.softmax(target_logits, dim=-1)

            self._record_metrics(
                active_logits,
                target_logits,
                current_gt_logits,
                target_gt_logits,
                max_other_logits,
            )

        log_probs = F.log_softmax(active_logits, dim=-1)
        loss_sum = F.kl_div(log_probs, target_probs, reduction="sum")

        if self.metric_accumulator is not None:
            mode = "train" if self.model is None or self.model.training else "eval"
            self.metric_accumulator.add(
                mode,
                "target_kl",
                loss_sum,
                active_logits.new_tensor(active_logits.size(0)),
            )

        # During training, Transformers supplies the number of valid tokens in
        # the whole accumulated batch. Dividing by it makes gradient
        # accumulation equivalent to one large batch. Evaluation calls may not
        # supply it, in which case a local token mean is the correct reduction.
        if num_items_in_batch is None:
            denominator = active_mask.sum().to(device=loss_sum.device, dtype=loss_sum.dtype)
        else:
            denominator = torch.as_tensor(
                num_items_in_batch,
                device=loss_sum.device,
                dtype=loss_sum.dtype,
            )

        return loss_sum / denominator.clamp_min(1.0)


class SteeringMetricsCallback(TrainerCallback):
    """Inject accumulated steering diagnostics into Trainer logs."""

    def __init__(self, metric_accumulator):
        self.metric_accumulator = metric_accumulator
        self.trainer = None

    def bind_trainer(self, trainer):
        self.trainer = trainer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self.trainer is None:
            return

        mode = "eval" if any(key.startswith("eval_") for key in logs) else "train"
        metrics, max_metrics = self.metric_accumulator.pop(mode)
        prefix = "eval_steer_" if mode == "eval" else "train_steer_"

        logged_metrics = {}
        for name, (numerator, denominator) in metrics.items():
            numerator = self.trainer.accelerator.reduce(numerator, reduction="sum")
            denominator = self.trainer.accelerator.reduce(denominator, reduction="sum")
            logged_metrics[f"{prefix}{name}"] = (
                numerator / denominator.clamp_min(1.0)
            ).item()
        for name, value in max_metrics.items():
            value = self.trainer.accelerator.reduce(value, reduction="max")
            logged_metrics[f"{prefix}{name}"] = value.item()

        logs.update(logged_metrics)
        # Trainer appends to log_history before firing on_log callbacks. Keep
        # the persisted state in sync with the metrics added above.
        if logged_metrics and state.log_history:
            state.log_history[-1].update(logged_metrics)


class BaseFineTuneKLCallback(TrainerCallback):
    """Measure full-vocabulary output drift on fixed assistant-token positions."""

    def __init__(self, probe_samples=32, max_tokens_per_batch=32):
        if max_tokens_per_batch < 0:
            raise ValueError("max_tokens_per_batch cannot be negative")
        self.probe_samples = probe_samples
        self.max_tokens_per_batch = max_tokens_per_batch
        self.trainer = None
        self.probe_dataset = None
        self.reference_log_probs = []
        self.reference_stats = {}

    def bind_trainer(self, trainer):
        self.trainer = trainer

    @staticmethod
    def _add_metric(metrics, name, numerator, denominator):
        numerator = numerator.detach().float()
        denominator = torch.as_tensor(
            denominator,
            device=numerator.device,
            dtype=torch.float32,
        ).detach()
        if name in metrics:
            old_numerator, old_denominator = metrics[name]
            metrics[name] = (
                old_numerator + numerator,
                old_denominator + denominator,
            )
        else:
            metrics[name] = (numerator, denominator)

    def _add_logit_stats(self, metrics, prefix, logits):
        if logits.numel() == 0:
            return
        num_tokens = logits.new_tensor(logits.size(0))
        self._add_metric(
            metrics, f"{prefix}_logit_mean", logits.sum(), logits.numel()
        )
        self._add_metric(
            metrics,
            f"{prefix}_top1_logit",
            logits.max(dim=-1).values.sum(),
            num_tokens,
        )
        self._add_metric(
            metrics,
            f"{prefix}_bottom1_logit",
            logits.min(dim=-1).values.sum(),
            num_tokens,
        )

    def _reduce_metrics(self, totals):
        metrics = {}
        for name, (numerator, denominator) in totals.items():
            numerator = self.trainer.accelerator.reduce(numerator, reduction="sum")
            denominator = self.trainer.accelerator.reduce(
                denominator, reduction="sum"
            )
            metrics[f"eval_probe_{name}"] = (
                numerator / denominator.clamp_min(1.0)
            ).item()
        return metrics

    def _get_probe_dataset(self):
        dataset = self.trainer.eval_dataset
        if isinstance(dataset, dict):
            dataset = next(iter(dataset.values()))
        if self.probe_samples <= 0:
            return dataset
        num_samples = min(len(dataset), self.probe_samples)
        return dataset.select(range(num_samples))

    def _get_active_logits(self, model, inputs):
        inputs = self.trainer._prepare_inputs(inputs)
        labels = inputs["labels"]
        model_inputs = {
            key: value
            for key, value in inputs.items()
            if key not in {"labels", "shift_labels", "_prediction_loss_only"}
        }
        model_inputs["use_cache"] = False

        with self.trainer.compute_loss_context_manager():
            outputs = model(**model_inputs)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

        shift_logits = logits[..., :-1, :]
        shift_labels = labels[..., 1:]
        active_logits = shift_logits.reshape(-1, shift_logits.size(-1))[
            shift_labels.reshape(-1).ne(-100)
        ]
        if (
            self.max_tokens_per_batch > 0
            and active_logits.size(0) > self.max_tokens_per_batch
        ):
            # Cover the whole answer instead of biasing the probe toward its
            # first tokens. The selection is deterministic across all passes.
            selected = torch.linspace(
                0,
                active_logits.size(0) - 1,
                steps=self.max_tokens_per_batch,
                device=active_logits.device,
            ).round().long()
            active_logits = active_logits.index_select(0, selected)
        return active_logits.float()

    def capture_base(self):
        if self.trainer is None:
            return

        self.probe_dataset = self._get_probe_dataset()
        dataloader = self.trainer.get_eval_dataloader(self.probe_dataset)
        model = self.trainer.model_wrapped
        was_training = model.training
        model.eval()
        reference_stats = {}
        self.reference_log_probs = []

        with torch.no_grad():
            for inputs in dataloader:
                active_logits = self._get_active_logits(model, inputs)
                self._add_logit_stats(reference_stats, "base", active_logits)
                base_log_probs = F.log_softmax(active_logits, dim=-1)
                # Keep every vocabulary entry. fp16 CPU storage controls RAM;
                # values are restored to fp32 and re-normalized for KL.
                self.reference_log_probs.append(
                    base_log_probs.to(device="cpu", dtype=torch.float16)
                )

        model.train(was_training)
        self.reference_stats = {
            name: (numerator.cpu(), denominator.cpu())
            for name, (numerator, denominator) in reference_stats.items()
        }

        if self.trainer.is_world_process_zero():
            stored_tokens = sum(
                log_probs.size(0) for log_probs in self.reference_log_probs
            )
            print(
                f"[KL probe] Captured full-vocabulary base distributions for "
                f"{stored_tokens} assistant-token positions on this process."
            )

        # Add a zero-drift point at global step 0 without running the model a
        # second time. At this point the fine-tuned distribution is the base
        # distribution by construction.
        device = self.trainer.args.device
        baseline_totals = {
            name: (numerator.to(device), denominator.to(device))
            for name, (numerator, denominator) in self.reference_stats.items()
        }
        baseline_metrics = self._reduce_metrics(baseline_totals)
        for suffix in ("logit_mean", "top1_logit", "bottom1_logit"):
            baseline_metrics[f"eval_probe_finetuned_{suffix}"] = baseline_metrics[
                f"eval_probe_base_{suffix}"
            ]
        baseline_metrics["eval_probe_kl_base_to_ft"] = 0.0
        baseline_metrics["eval_probe_kl_ft_to_base"] = 0.0
        self.trainer.log(baseline_metrics)

    def _compute_metrics(self):
        if not self.reference_log_probs:
            return {}

        dataloader = self.trainer.get_eval_dataloader(self.probe_dataset)
        model = self.trainer.model_wrapped
        was_training = model.training
        model.eval()
        device = self.trainer.args.device
        totals = {
            name: (numerator.to(device), denominator.to(device))
            for name, (numerator, denominator) in self.reference_stats.items()
        }

        with torch.no_grad():
            for batch_index, inputs in enumerate(dataloader):
                if batch_index >= len(self.reference_log_probs):
                    raise RuntimeError("KL probe dataloader produced more batches than the base pass")

                current_logits = self._get_active_logits(model, inputs)
                base_log_probs = self.reference_log_probs[batch_index].to(
                    device=current_logits.device, dtype=torch.float32
                )
                if base_log_probs.shape != current_logits.shape:
                    raise RuntimeError(
                        "KL probe token shape changed between base and fine-tuned passes"
                    )

                # Re-normalize after fp16 CPU storage before computing KL over
                # the complete vocabulary.
                base_log_probs = base_log_probs - torch.logsumexp(
                    base_log_probs, dim=-1, keepdim=True
                )
                current_log_probs = F.log_softmax(current_logits, dim=-1)

                kl_base_to_ft = F.kl_div(
                    current_log_probs,
                    base_log_probs,
                    log_target=True,
                    reduction="none",
                ).sum(dim=-1)
                kl_ft_to_base = F.kl_div(
                    base_log_probs,
                    current_log_probs,
                    log_target=True,
                    reduction="none",
                ).sum(dim=-1)

                num_tokens = current_logits.new_tensor(current_logits.size(0))
                self._add_metric(
                    totals,
                    "kl_base_to_ft",
                    kl_base_to_ft.sum(),
                    num_tokens,
                )
                self._add_metric(
                    totals,
                    "kl_ft_to_base",
                    kl_ft_to_base.sum(),
                    num_tokens,
                )
                self._add_logit_stats(totals, "finetuned", current_logits)

        model.train(was_training)
        return self._reduce_metrics(totals)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.trainer is None or not self.reference_log_probs:
            return
        probe_metrics = self._compute_metrics()
        if metrics is not None:
            metrics.update(probe_metrics)
        self.trainer.log(probe_metrics)
        if self.trainer.is_world_process_zero():
            save_monitoring_plots(state.log_history, args.output_dir)


def build_steered_loss(x_factor, metric_accumulator=None):
    """Build a Trainer-compatible steered loss function."""
    return SteeredLoss(x_factor, metric_accumulator=metric_accumulator)


def save_monitoring_plots(log_history, output_dir):
    """Save step-indexed KL and steering diagnostic charts."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Monitoring] matplotlib is unavailable; skipping PNG plots.")
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    def series(key):
        points = [
            (entry["step"], entry[key])
            for entry in log_history
            if "step" in entry and key in entry
        ]
        return zip(*points) if points else ([], [])

    def plot_line(axis, key, label, **kwargs):
        steps, values = series(key)
        if not steps:
            return False
        axis.plot(steps, values, marker="o", markersize=3, label=label, **kwargs)
        return True

    drift_keys = {
        "kl": "eval_probe_kl_base_to_ft",
        "reverse_kl": "eval_probe_kl_ft_to_base",
        "base_mean": "eval_probe_base_logit_mean",
        "ft_mean": "eval_probe_finetuned_logit_mean",
        "base_top1": "eval_probe_base_top1_logit",
        "ft_top1": "eval_probe_finetuned_top1_logit",
        "base_bottom1": "eval_probe_base_bottom1_logit",
        "ft_bottom1": "eval_probe_finetuned_bottom1_logit",
    }
    if any(any(key in entry for entry in log_history) for key in drift_keys.values()):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        plot_line(axes[0, 0], drift_keys["kl"], "KL(base || fine-tuned)")
        plot_line(
            axes[0, 0],
            drift_keys["reverse_kl"],
            "KL(fine-tuned || base)",
        )
        axes[0, 0].set_title("Full-vocabulary output-distribution drift")

        for axis, title, base_key, ft_key in [
            (axes[0, 1], "Mean vocabulary logit", "base_mean", "ft_mean"),
            (axes[1, 0], "Highest vocabulary logit", "base_top1", "ft_top1"),
            (axes[1, 1], "Lowest vocabulary logit", "base_bottom1", "ft_bottom1"),
        ]:
            plot_line(axis, drift_keys[base_key], "Base")
            plot_line(axis, drift_keys[ft_key], "Fine-tuned")
            axis.set_title(title)

        for axis in axes.flat:
            axis.set_xlabel("Global training step")
            axis.grid(alpha=0.25)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend()
        drift_path = output_path / "base_finetune_drift.png"
        fig.savefig(drift_path, dpi=160)
        plt.close(fig)
        saved_paths.append(drift_path)

    steering_panels = [
        ("Average vocabulary logit", "logit_mean_before", "logit_mean_after"),
        ("Highest vocabulary logit", "top1_logit_before", "top1_logit_after"),
        ("Lowest vocabulary logit", "bottom1_logit_before", "bottom1_logit_after"),
        ("Ground-truth token logit", "gt_logit_before", "gt_logit_after"),
        ("Ground-truth margin gap", "margin_gap_before", "margin_gap_after"),
    ]
    has_steering_metrics = any(
        any(f"train_steer_{key}" in entry for entry in log_history)
        for _, before_key, after_key in steering_panels
        for key in (before_key, after_key)
    )
    if has_steering_metrics:
        fig, axes = plt.subplots(4, 2, figsize=(13, 15), constrained_layout=True)
        flat_axes = list(axes.flat)
        for axis, (title, before_key, after_key) in zip(flat_axes, steering_panels):
            plot_line(axis, f"train_steer_{before_key}", "Before margin")
            plot_line(axis, f"train_steer_{after_key}", "After margin")
            axis.set_title(title)

        plot_line(
            flat_axes[5],
            "train_steer_activated_fraction",
            "Fraction changed by margin",
        )
        flat_axes[5].set_title("Margin activation rate")
        flat_axes[5].set_ylim(-0.02, 1.02)
        plot_line(flat_axes[6], "train_steer_target_kl", "Steering target KL")
        flat_axes[6].set_title("Mean steering target KL per token")
        plot_line(
            flat_axes[7],
            "train_steer_gt_logit_boost_activated_mean",
            "Mean boost",
        )
        plot_line(
            flat_axes[7],
            "train_steer_gt_logit_boost_activated_max",
            "Maximum boost",
        )
        flat_axes[7].set_title("Extra ground-truth logit when steered")

        for axis in flat_axes:
            axis.set_xlabel("Global training step")
            axis.grid(alpha=0.25)
            handles, _ = axis.get_legend_handles_labels()
            if handles:
                axis.legend()
        steering_path = output_path / "steering_margin_diagnostics.png"
        fig.savefig(steering_path, dpi=160)
        plt.close(fig)
        saved_paths.append(steering_path)

    for saved_path in saved_paths:
        print(f"[Monitoring] Saved chart: {saved_path}")
    return saved_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Run Steered SFT Training")
    parser.add_argument("--model_name_or_path", type=str, default=QWEN2_BASE_MODEL_ID)
    parser.add_argument("--train_data_path", type=str, default="data/medqa_data/train_data")
    parser.add_argument("--eval_data_path", type=str, default="data/medqa_data/eval_data")
    parser.add_argument("--dev_data_path", type=str, default="data/medqa_data/dev_data")
    parser.add_argument("--output_dir", type=str, default="outputs/steer-qwen2-7b")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--x_factor", type=float, default=1.0)
    parser.add_argument(
        "--kl_probe_samples",
        type=int,
        default=32,
        help="Fixed dev samples used for full-vocabulary output KL; 0 uses all",
    )
    parser.add_argument(
        "--kl_probe_max_tokens_per_batch",
        type=int,
        default=32,
        help="Evenly sampled assistant-token positions per probe batch; 0 keeps all",
    )
    parser.add_argument(
        "--disable_kl_monitor",
        action="store_true",
        help="Disable base-vs-finetuned output-distribution KL monitoring",
    )
    parser.add_argument("--disable_peft", action="store_true", help="Vô hiệu hóa LoRA (Full Fine-Tuning)")
    return parser.parse_args()

def prepare_train_dataset(dataset_path):
    print(f"Loading train dataset from {dataset_path}")
    dataset = Dataset.load_from_disk(dataset_path)
    
    def format_messages(example):
        msgs = example.get("messages")
        output_text = str(example["output_text"])

        if isinstance(msgs, list) and msgs:
            msgs_copy = [dict(message) for message in msgs]
            if msgs_copy[-1].get("role") == "assistant":
                if str(msgs_copy[-1].get("content", "")) != output_text:
                    raise ValueError("The final assistant message does not match output_text")
            else:
                msgs_copy.append({"role": "assistant", "content": output_text})
            example["messages"] = msgs_copy
        else:
            example["messages"] = [
                {"role": "user", "content": str(example.get("prompt", ""))},
                {"role": "assistant", "content": output_text},
            ]
        return example
        
    dataset = dataset.map(format_messages, desc="Formatting conversation messages")
    
    cols_to_remove = [col for col in ["prompt", "text", "output_text"] if col in dataset.column_names]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)
        
    return dataset

def main():
    args = parse_args()
    
    train_dataset = prepare_train_dataset(args.train_data_path)
    eval_dataset = prepare_train_dataset(args.dev_data_path)
    
    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_length=args.max_seq_length,
        assistant_only_loss=True,
        # A custom logits-based loss needs the regular LM head output. Keeping
        # TRL's default chunked_nll would patch a labels-only path that this
        # compute_loss_func cannot use.
        loss_type="nll",
        model_init_kwargs={"dtype": "bfloat16"},
        average_tokens_across_devices=True,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        weight_decay=0.1,
        max_grad_norm=1.0,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=0.2,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        report_to="none",
        bf16=True,
        fp16=False,
        gradient_checkpointing=True,
        optim="adamw_torch"
    )
    
    peft_config = None
    if not args.disable_peft:
        peft_config = LoraConfig(
            r=128,
            lora_alpha=256,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
            bias="none",
        )
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    configured_qwen2_base = configure_qwen2_base_tokenizer(
        tokenizer, args.model_name_or_path
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if configured_qwen2_base:
        print(
            "[Qwen2 base] Enabled assistant-only ChatML masking; "
            "EOS=<|im_end|>, PAD=<|endoftext|>."
        )

    steer_metric_accumulator = WeightedMetricAccumulator()
    steer_metrics_callback = SteeringMetricsCallback(steer_metric_accumulator)
    steered_loss = build_steered_loss(
        args.x_factor,
        metric_accumulator=steer_metric_accumulator,
    )

    trainer = SFTTrainer(
        model=args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        compute_loss_func=steered_loss,
        callbacks=[steer_metrics_callback],
    )

    if configured_qwen2_base:
        sync_model_special_tokens(trainer.model, tokenizer)

    steered_loss.bind_model(trainer.model)
    steer_metrics_callback.bind_trainer(trainer)

    if not args.disable_kl_monitor:
        kl_callback = BaseFineTuneKLCallback(
            probe_samples=args.kl_probe_samples,
            max_tokens_per_batch=args.kl_probe_max_tokens_per_batch,
        )
        kl_callback.bind_trainer(trainer)
        trainer.add_callback(kl_callback)
        kl_callback.capture_base()

    print("\nBắt đầu quá trình huấn luyện Steered SFT...")
    trainer.train()
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    if trainer.is_world_process_zero():
        save_monitoring_plots(trainer.state.log_history, args.output_dir)
    import shutil, glob
    for ckpt in glob.glob(f"{args.output_dir}/checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)
    print(f"Huấn luyện hoàn tất! Model lưu tại: {args.output_dir}")

if __name__ == "__main__":
    main()
