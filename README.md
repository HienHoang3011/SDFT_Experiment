# Self-Distilled Fine-Tuning (SDFT) Pipeline

This project implements the Self-Distilled Fine-Tuning (SDFT) approach for continual learning without catastrophic forgetting.

## Dataset Information

- **Dataset:** GBaker/MedQA-USMLE-4-options
- **Train Data:** 10,178 examples (Used for training. During training, it is automatically split into ~9,160 train / ~1,018 validation examples).
- **Eval Data (Test):** 1,273 examples (Used strictly as a held-out test set to evaluate New Task Accuracy).

## Installation

Ensure you have installed all required dependencies in your environment using `uv` (Note: `vllm` is excluded for Windows compatibility):

```bash
uv add datasets trl peft matplotlib lm-eval transformers torch
```

> [!NOTE]
> **Troubleshooting GPU/CUDA Issues:**
> If you encounter errors like `ValueError: Your setup doesn't support bf16/gpu` or warnings about the NVIDIA driver being too old (e.g., driver version 525.xx / CUDA 12.0) during training, PyTorch may not be recognizing your GPU due to a CUDA version mismatch. 
> 
> You can fix this by reinstalling PyTorch with a compatible CUDA version (e.g., CUDA 11.8) using:
> ```bash
> uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
> ```

## Pipeline Execution

To run the entire end-to-end pipeline, which includes:
1. Training models using different approaches: Full SFT, SFT LoRA, FAPM, Steered SFT, and Steered SFT Full.
2. Evaluating the MedQA Task (New Task) and Prior Capabilities (Old Tasks) for the Base Model.
3. Evaluating the MedQA Task and Prior Capabilities for all the trained models (SFT, SFT LoRA, FAPM, Steered SFT, Steered SFT Full).
4. Generating a comparative report and visualization plot across all methods.

Run the following command:

```bash
uv run python main.py
```

### Optional Arguments
You can specify a different base model or output directory when running the pipeline:

```bash
uv run python main.py --base_model "Qwen/Qwen2-7B"
```

## Generated Reports

After the pipeline finishes, the results will be saved in the `reports/` directory:
- A printed markdown table comparing "New Task Accuracy" vs "Prior Tasks Performance".
- `reports/sdft_performance_plot.png`: A scatter plot visualization showing the performance trade-off between the Base Model and SDFT.

# Qwen2 catastrophic-forgetting screening

Run the complete data preparation, validation, Full SFT, LoRA, downstream
evaluation, prior-capability evaluation, and reporting pipeline with:

```bash
bash run_cf_screening.sh
```

The default tasks are WinoGrande, SQuAD, and RTE. Reports are written to
`reports-winogrande`, `reports-squad`, and `reports-rte`. Each report contains
the data-quality audit, raw downstream predictions, retained-capability
results, a Markdown/JSON table, `results_table.png`,
`performance_plot_5_models.png`, and `prior_tasks_breakdown.png`, matching the
report family produced by the original pipeline.

Full-SFT and LoRA model directories are temporary. Each model is trained,
evaluated on both downstream and prior tasks, and then deleted; reports and raw
predictions remain. Set `KEEP_MODELS=1` only when checkpoints are needed for
debugging.

Training calls the existing `methods/train_sft.py` and
`methods/train_sft_lora.py` files. Defaults shared by all training methods are
the disclosed FAPM settings: learning rate `1e-5`, effective batch size `64`
for the default one-process/batch-one run, three epochs, AdamW, cosine
scheduling, warmup ratio `0.03`, weight decay `0.1`, and sequence length
`2048`. LoRA-only fields not disclosed by FAPM remain the repository defaults:
rank `128`, alpha `256`, and dropout `0.05`.

Prior evaluation is enabled and uses HellaSwag, MMLU, TruthfulQA, HumanEval,
and IFEval. WinoGrande is deliberately excluded because it is a downstream
training task in this screening suite.

Environment variables can override the run without editing the script:

```bash
TASKS="winogrande rte" EPOCHS=1 bash run_cf_screening.sh
```

For a small smoke run, use separate paths so the reduced datasets are not
reused by a later full run:

```bash
DATA_ROOT=data/cf_benchmarks-smoke \
OUTPUT_ROOT=outputs/cf-screening-smoke \
MAX_TRAIN_SAMPLES=64 MAX_EVAL_SAMPLES=32 DOWNSTREAM_EVAL_LIMIT=32 \
bash run_cf_screening.sh
```
