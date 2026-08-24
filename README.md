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
uv run python main.py --base_model "Qwen/Qwen2.5-7B-Instruct" --output_model_dir "outputs/sdft-qwen2.5-7b"
```

## Generated Reports

After the pipeline finishes, the results will be saved in the `reports/` directory:
- A printed markdown table comparing "New Task Accuracy" vs "Prior Tasks Performance".
- `reports/sdft_performance_plot.png`: A scatter plot visualization showing the performance trade-off between the Base Model and SDFT.
