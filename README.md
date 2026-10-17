# SDFT Experiment: Steer, FAPM và quên kiến thức khi fine-tune

Fine-tune Qwen2-7B trên SAMSum (tóm tắt hội thoại) và đo mức quên năng lực chung bằng 5 benchmark
(GSM8K, HumanEval, IFEval, BoolQ, BBH). So sánh CE thường (Full SFT, LoRA) với Steer, một loss self-margin KL,
và FAPM, một phương pháp hậu xử lý task vector.

## Trạng thái (2026-10-14)

- Full SFT, Steer-Full và FAPM của lượt đầu bị lỗi train thuần bf16 và đã được chạy lại với weight fp32
  ([05](docs/experiments/05-bf16-diagnosis-fp32-rerun.md)); cả sáu model đã chấm lại trong cùng môi trường.
- Phân tích vì sao model quên: [06](docs/experiments/06-forgetting-analysis.md).
- Base, LoRA và Steer-LoRA không bị ảnh hưởng; kết quả ở [01](docs/experiments/01-six-method-comparison.md) vẫn dùng được.

## Kết quả chính

| | Base | LoRA | Steer-LoRA | Full SFT fp32 | Steer-Full fp32 | FAPM fp32 |
|---|---:|---:|---:|---:|---:|---:|
| SAMSum ROUGE-L | 33,06 | **46,27** | 44,06 | 44,89 | 43,73 | 41,27 |
| GSM8K | 70,74 | 60,96 | 63,23 | 38,89 | 32,90 | **66,03** |
| HumanEval | 70,12 | 65,24 | 61,59 | 59,76 | 58,54 | **67,07** |
| IFEval | 32,72 | **28,10** | 23,11 | 17,56 | 13,31 | 25,51 |
| BoolQ | 80,64 | **83,06** | 62,20 | 57,43 | 62,17 | 63,58 |
| BBH | 48,55 | **44,66** | 42,80 | 33,79 | 33,56 | 39,27 |

Cả sáu model chấm trong cùng một môi trường (2026-10-14); in đậm là model đã fine-tune tốt nhất ở mỗi hàng.
Phân tích vì sao: [benchmark-analysis](artifacts/samsum/fp32-master-2epoch-20261012/benchmark-analysis/README.md),
[FAPM](artifacts/samsum/fp32-master-2epoch-20261012/fapm-analysis/README.md),
[cơ chế forgetting](artifacts/samsum/fp32-master-2epoch-20261012/forgetting-mechanism/README.md).

- Steer là cross-entropy có trọng số theo token; nó bỏ qua token đã đạt margin và đẩy rất nhẹ ở token phân vân, nên làm phẳng
  phân phối thay vì làm sắc. Hệ quả rõ nhất: với prompt ChatML, BoolQ luôn trả lời "yes" (−21 điểm so với LoRA).
- FAPM ≈ Base ở lượt bf16 là do lỗi bf16, không phải do phương pháp; ở lượt fp32, FAPM giữ được phần lớn chất lượng tóm tắt.

## Tài liệu

| Tài liệu | Nội dung |
|---|---|
| [docs/setup.md](docs/setup.md) | Môi trường, dữ liệu, thiết lập train và đánh giá chung |
| [docs/methods.md](docs/methods.md) | Công thức Steer, FAPM, L1 |
| [docs/experiments/01-six-method-comparison.md](docs/experiments/01-six-method-comparison.md) | So sánh sáu phương pháp, 2 epoch |
| [docs/experiments/02-fixed-margin-controls.md](docs/experiments/02-fixed-margin-controls.md) | Sáu đối chứng 128 bước |
| [docs/experiments/03-steer-base-anchor.md](docs/experiments/03-steer-base-anchor.md) | Steer có neo Base |
| [docs/experiments/04-update-directions.md](docs/experiments/04-update-directions.md) | Hướng cập nhật từ checkpoint |
| [docs/experiments/05-bf16-diagnosis-fp32-rerun.md](docs/experiments/05-bf16-diagnosis-fp32-rerun.md) | Lỗi bf16 và lượt chạy lại fp32 |
| [docs/experiments/06-forgetting-analysis.md](docs/experiments/06-forgetting-analysis.md) | Vì sao model quên: định dạng, lớp, gradient, FAPM |
| [docs/artifacts.md](docs/artifacts.md) | Bố cục `artifacts/` |
| [docs/changelog.md](docs/changelog.md) | Lịch sử thay đổi code và artifact, vấn đề đã biết |

## Cấu trúc source

| Thư mục | Nội dung |
|---|---|
| `methods/` | Script train (`train_sft.py`, `train_sft_lora.py`, `train_steer.py`, `train_fixed_margin_controls.py`, `train_l1.py`), `fapm.py`, `base_anchored_steer.py`, `qwen2_base.py` |
| `eval/` | Đánh giá (`eval_samsum.py`, `eval_kl.py`, `analyze_boolq_margin.py`), metric dùng chung, script phân tích, `upload_models_wandb.py` |
| `benchmarks/stable_five/` | Runner 5 benchmark lm-eval và script vẽ bảng điểm |
| `tests/` | Test CPU: `python -m pytest tests` |
| `run_fp32_master_rerun.sh` | Pipeline chạy lại Full SFT, Steer-Full, FAPM với weight fp32 |
| `prepare_samsum_emnlp2019.py` | Chuẩn bị dữ liệu SAMSum |

Kết quả và checkpoint nằm trong `artifacts/` (ngoài git). Model lưu trữ trên W&B: project `sdft-experiment`.
