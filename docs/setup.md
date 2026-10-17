# Môi trường, dữ liệu và thiết lập chung

## Mô hình và dữ liệu

- Mô hình gốc: Qwen2-7B base, checkpoint `artifacts/base/Qwen2-7B`.
- Dữ liệu: SAMSum (EMNLP 2019) từ mirror `fastlayne/samsum`, đã bỏ mẫu rỗng và hội thoại trùng giữa các split.
  Theo manifest: **14.639 train, 814 dev, 819 test**, lưu ở `data/samsum_emnlp2019/{train,dev,test}`.
- Prompt yêu cầu tóm tắt trong `<answer>…</answer>`; chỉ tính loss trên token assistant (ChatML, `methods/qwen2_base.py`).
- Chuẩn bị dữ liệu trên workspace mới: `python prepare_samsum_emnlp2019.py --output_root data/samsum_emnlp2019`.

## Môi trường Python

Môi trường của project là `.venv/`, tạo lại ngày 2026-10-13 cho khớp đúng môi trường đã chạy lượt fp32
(`/root/sdft-fp32-runs/venv`). Cả hai dùng lại torch CUDA của hệ thống qua `--system-site-packages`.
Phiên bản được ghi trong `env/requirements-runtime.txt`; tạo lại bằng
`python3 -m venv --system-site-packages .venv && .venv/bin/pip install -r env/requirements-runtime.txt`.

| Gói | Phiên bản |
|---|---|
| torch | 2.7.0a0 (bản NVIDIA của container) |
| transformers / trl / peft | 4.57.6 / 1.8.0 / 0.19.1 |
| bitsandbytes / accelerate / datasets | 0.50.0 / 1.15.0 / 5.0.0 |
| lm-eval | 0.4.12 (cần `typing_extensions >= 4.13`) |
| pyarrow / numpy / rouge-score | 25.0.1 / 1.26.4 / 0.1.2 |

`uv.lock` và `pyproject.toml` đã lỗi thời so với môi trường thật (khai báo torch 2.6.0+cu124 và numpy 2.2.6; máy không
cài uv) nên chưa dùng được để dựng lại môi trường. Môi trường cũ `/tmp/sdft-eval-venv` đã mất. Mỗi lượt chạy lưu
`pip freeze` ở `logs/environment.txt`.

Chạy test CPU: `.venv/bin/python -m pytest tests` (33 test).

## Thiết lập train chung

| Tham số | Giá trị |
|---|---|
| Epoch | 2 (458 bước optimizer) |
| Learning rate | 1e-5, cosine, warmup 3% |
| Batch | 1 mỗi GPU × gradient accumulation 64 = 64 |
| Max sequence length | 2048 |
| Optimizer | AdamW 8-bit (`paged_adamw_8bit` cho full fine-tune fp32, cùng công thức cập nhật) |
| Weight decay / clip | 0.1 / 1.0 |
| Seed / data seed | 42 / 42 |
| LoRA | rank 256, alpha 512, dropout 0,3; q/k/v/o/gate/up/down |
| Độ chính xác | Tính toán bf16 (autocast). Full fine-tune lưu weight fp32 từ 2026-10-12, xem [experiments/05](experiments/05-bf16-diagnosis-fp32-rerun.md) |
| Checkpoint | Chỉ lưu model cuối |

## Đánh giá

| Công cụ | Thiết lập |
|---|---|
| `eval/eval_samsum.py` | 819 mẫu test, greedy, `max_new_tokens=128`, ROUGE-L (rouge-score, stemmer) |
| `benchmarks/stable_five/run.py` | lm-eval 0.4.12: GSM8K (1.319, CoT zero-shot, 1.024 token), HumanEval (164, pass@1), IFEval (541, prompt-level strict), BoolQ (3.270, acc), BBH (6.511, CoT zero-shot, 2.048 token). Chat template bật cho mọi model. |
| `eval/eval_kl.py` | Phân phối teacher-forced trên SAMSum dev: hai chiều KL, JS, TV, entropy, NLL; `--margin-diagnostics` thêm số liệu margin của Steer |
| `eval/analyze_boolq_margin.py` | BoolQ raw và ChatML, so log-prob token no/yes |

## Chạy dài: dùng tmux

Mọi tiến trình dài chạy trong tmux session `sdft` để không phụ thuộc phiên terminal:
`tmux attach -t sdft`, thoát ra bằng `Ctrl-b d`. Script khởi chạy: `/root/sdft-fp32-runs/tmux_pipeline.sh`,
`/root/sdft-fp32-runs/tmux_upload.sh`.
