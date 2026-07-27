# Self-Distilled Fine-Tuning (SDFT) Pipeline

This project implements the Self-Distilled Fine-Tuning (SDFT) approach for continual learning without catastrophic forgetting.

## Dataset Information

- **Train Data:** 2,674 examples (Used for training. During training, it is automatically split into ~2,139 train / ~535 validation examples).
- **Eval Data (Test):** 507 examples (Used strictly as a held-out test set to evaluate New Task Accuracy).

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
1. Training the model using SDFT.
2. Evaluating the Science Task (New Task) and Prior Capabilities (Old Tasks) for the Base Model.
3. Evaluating the Science Task and Prior Capabilities for the SDFT Finetuned Model.
4. Generating a comparative report and visualization plot.

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
