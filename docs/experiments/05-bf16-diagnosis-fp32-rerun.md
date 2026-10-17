# 05 · Lỗi bf16 thuần ở full fine-tune và lượt chạy lại fp32

**Trạng thái:** chẩn đoán xong 2026-10-12; lượt chạy lại và chấm lại cả sáu model xong 2026-10-14.
Artifact: `artifacts/samsum/fp32-master-2epoch-20261012/`; model ở `/root/sdft-fp32-runs/models/` (ổ `/data` không đủ chỗ).

## Vấn đề

Full SFT (`train_sft.py`) và Steer-Full (`train_steer.py --disable_peft`) load model ở bf16 và không có bản weight fp32.
Ở |w| ∈ [2⁻⁸, 2⁻⁷), hai số bf16 liền kề cách nhau 2⁻¹⁵; update AdamW khoảng 1e-5 nhỏ hơn nửa khoảng đó nên bị làm tròn mất.
Mixed precision chuẩn (tính bf16, weight fp32) không có vấn đề này. LoRA không bị ảnh hưởng: adapter được PEFT giữ fp32
(kiểm chứng: 100% giá trị A/B lệch khỏi lưới bf16).

## Số liệu chẩn đoán (quét đủ 339/339 tensor, 7.615.616.512 tham số)

| \|w\| của Base | Full SFT bf16 đã đổi |
|---|---:|
| < 2⁻¹⁰ | 77–91% |
| [2⁻¹⁰, 2⁻⁹) | 74% |
| [2⁻⁹, 2⁻⁸) | 9% |
| [2⁻⁸, 2⁻⁷) | 0,04% |
| ≥ 2⁻⁷ | ≈ 0% |

- Full SFT bf16 chỉ đổi 11,46% tham số; 54% weight lớn nhất không đổi sau 2 epoch. Dù vậy vẫn fit SAMSum ngang LoRA
  (ΔNLL dev −0,516 so với −0,520).
- Gradient CE tại Base (64 ví dụ, 2.097 token): 63% năng lượng (tổng bình phương) nằm trên weight |w| ≥ 2⁻⁸;
  chỉ 19,5% nằm trên các phần tử Full SFT đã đổi. cos(Δ, −gradient): Full SFT 0,036, Steer-Full 0,029, FAPM ≈ 0.
- **FAPM ≈ Base là hệ quả:** 99,985% phần tử Full SFT đã đổi có |w| < λ nên điểm t < 0; phần tử không đổi có t = 0
  và chiếm top-k. FAPM chỉ đổi 129.285 tham số (0,0017%), giữ 1,74% năng lượng Δ; KL(Base ‖ FAPM) = 0,0008.
- Quét k_ratio (k = 0,1 dựng lại khớp từng bit với FAPM đã lưu):

  | k | Tham số khác Base | Năng lượng Δ giữ | ROUGE-L | BoolQ ChatML |
  |---:|---:|---:|---:|---:|
  | 10% | 129.285 | 1,74% | 32,98 | 80,64 |
  | 50% | 129.899 | 1,74% | 32,88 | 80,49 |
  | 80% | 10,4 triệu | 1,80% | — | — |
  | 90% | 163,9 triệu | 3,21% | 35,05 | 80,00 |

Thư mục phân tích gốc (`artifacts/samsum/tensor-change-analysis-20261012/`, gồm heatmap theo lớp × module) đã bị xoá
ngày 2026-10-13; các số liệu trên được ghi lại từ đó. Checkpoint bf16 cũ cũng đã xoá nên không chạy lại được.

## Sửa code (2026-10-12)

- `train_sft.py`: `--model_dtype` (mặc định `float32`); `train_steer.py`: `--full_ft_dtype` (mặc định `float32`, chỉ khi
  `--disable_peft`); `train_l1.py`: `--model_dtype`. `bf16=True` vẫn bật autocast cho phần tính toán.
- `fapm.py`: load SFT ở fp32, tính Δ và t ở fp32, chỉ ép kết quả merge về bf16.
- Với weight fp32, `adamw_8bit` hết bộ nhớ trên H100 80 GB ở bước 2; dùng `paged_adamw_8bit` (cùng công thức cập nhật,
  trạng thái optimizer được phân trang sang RAM).

## Lượt chạy lại

Pipeline: `run_fp32_master_rerun.sh` (bỏ qua bước đã có output). Chạy trong tmux qua `/root/sdft-fp32-runs/tmux_pipeline.sh`.

| Bước | Trạng thái |
|---|---|
| Train Full SFT, Steer-Full (fp32, 458 bước, mỗi model khoảng 84 phút) | Xong |
| FAPM, SAMSum ROUGE-L, phân phối SAMSum dev, BoolQ raw/ChatML | Xong |
| 5 benchmark lm-eval cho Full SFT, Steer-Full, FAPM | Xong |
| Chấm lại Base, LoRA, Steer-LoRA (ROUGE-L + 5 benchmark) trong cùng môi trường | Xong |

Điểm cũ của Base, LoRA, Steer-LoRA được chấm bằng một môi trường đã mất (`/tmp/sdft-eval-venv`); chấm lại để cả sáu model
dùng chung một môi trường. Mọi benchmark áp chat template ChatML cho cả sáu model, kể cả Base. Đây là lựa chọn để thống
nhất định dạng, không phải lỗi, nhưng điểm tuyệt đối phụ thuộc định dạng (BoolQ của Steer đặc biệt nhạy).

Kết quả đến giờ:

| | Full SFT | Steer-Full | FAPM |
|---|---:|---:|---:|
| ROUGE-L, bf16 cũ | 46,04 | 43,69 | 32,93 |
| ROUGE-L, fp32 mới | 44,89 | 43,73 | **41,27** |

- Bản fp32 học thuộc dữ liệu ở epoch 2: train loss của Full SFT giảm từ 0,88 xuống khoảng 0,43, eval loss tăng từ
  0,800 lên 0,896. Bản bf16 không bị (nó gần như không thay đổi được weight).
- FAPM có tác dụng thật khi Full SFT thay đổi được weight lớn: ROUGE-L tăng từ 32,93 lên 41,27.

- Vì sao FAPM tốt (`fapm-analysis/README.md`, `eval/analyze_fapm_masks.py`): lợi ích chủ yếu do thu nhỏ task vector
  (‖Δ_FAPM‖ ≈ 0,51‖Δ‖). Nhân cả Δ với 0,51 cho GSM8K ngang FAPM (68,4 so với 69,2) mà ROUGE-L 44,9 (FAPM 41,5); top-10% theo
  \|Δ\| cũng ≥ FAPM. GSM8K của Full SFT giảm chủ yếu vì trả lời theo định dạng `<answer>` của SAMSum.

- Phân tích từng mẫu (`benchmark-analysis/README.md`): Full SFT/Steer-Full bọc 95–100% câu GSM8K/BBH trong `<answer>`;
  FAPM trả lời rỗng ở 17% IFEval, 8% BBH; BoolQ ChatML đo thiên lệch yes/no chứ không đo kiến thức.
- Cơ chế forgetting (`forgetting-mechanism/README.md`, `eval/analyze_forgetting_mechanism.py`):
  - KL(Base ‖ model) trên lời giải của Base tập trung ở token đầu (Full SFT 14,7 so với 0,23 ở các token sau); mồi
    "Let's solve this step by step." đưa GSM8K của Full SFT từ 40,4 lên 57,2.
  - Định dạng `<answer>` được quyết định ở lớp muộn (logit lens lớp 24–27; chỉ giữ Δ ở lớp 21–27 đã đủ 20/20 câu mở thẻ).
    Chỉ giữ Δ ở lớp 0–20 cho ROUGE-L 44,9 nhưng model không dừng được.
  - Gradient SAMSum và gradient task chung gần vuông góc (cos −0,006) trừ các vector norm (norm cuối −0,90).
    Update của Steer đi ngược gradient task chung nhiều hơn CE (cos −0,018 so với −0,006 ở bản full).
  - FAPM gõ `<|im_end|>` thành chữ thường vì mask 10% làm update giữa các lớp không khớp (trả lại update cho hàng
    token đặc biệt không chữa được; trả đủ Δ cho lớp muộn hoặc lớp sớm thì chữa được).
- Quét α·Δ: α = 0,5 cho ROUGE-L 45,3 và GSM8K 66,8; GSM8K giảm 20 điểm giữa α = 0,75 và 1 (dạng ngưỡng).

Hình: `training_curves_full_sft.png`, `training_curves_steer_full.png` (vẽ bằng `plot_training_curves.py` trong artifact);
bảng điểm `scores-fp32-rerun.png`, `scores-bf16-vs-fp32.png` (vẽ bằng `benchmarks/stable_five/plot_fp32_rerun.py`).

## Lưu trữ model trên W&B

Project `hoangchihien301105-personal-/sdft-experiment`, upload bằng `eval/upload_models_wandb.py` (model full được chuyển
sang bf16 trước khi upload; chỉ xoá bản local khi danh sách file trên W&B khớp). Không upload Base (có trên Hugging Face)
và FAPM (tính lại được từ Full SFT). Hàng đợi: `/root/sdft-fp32-runs/tmux_upload.sh`, log ở `/root/sdft-fp32-runs/upload-logs/`.
