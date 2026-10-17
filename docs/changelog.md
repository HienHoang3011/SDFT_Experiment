# Thay đổi code và artifact

## 2026-10-13

- Xoá checkpoint bf16 cũ `full_sft`, `steer_full`, `fapm` trong `fair-methods-2epoch-log25-tmux/models/` (theo yêu cầu;
  điểm số và báo cáo giữ lại).
- Phát hiện các file sau đã bị xoá (không rõ nguyên nhân, không khôi phục theo quyết định của người dùng):
  `run_samsum_tmux.sh`, `run_samsum_stable.sh`, `run_samsum_eval.sh`, `run_fixed_margin_causal_study.sh`,
  `run_anchored_steer_experiment.sh`, `run_checkpoint_direction_study.sh`, `run_distribution_comparison.sh`,
  `ARTIFACTS.md`, `FIXED_MARGIN_EXPERIMENT.md`, `artifacts/samsum/steer-base-anchor-2epoch-lambda1-20261003/`,
  `artifacts/samsum/tensor-change-analysis-20261012/`. Lệnh thay thế cho các script shell nằm trong `docs/experiments/`.
- Gộp tài liệu vào `docs/`: `EXPERIMENTS.md`, `UPDATE_DIRECTION_EXPERIMENT.md`, `FAPM_REVIEW.md`, `CODE_CLEANUP.md` được
  thay bằng `docs/setup.md`, `docs/methods.md`, `docs/experiments/*.md`, `docs/artifacts.md` và file này.
- Viết lại `.gitignore`; xoá `.pytest_cache/`, `artifacts/hf-cache/`; chuyển file zip backup vào `artifacts/backups/`.
- Môi trường chạy: nâng `typing_extensions` lên 4.16 (lm-eval 0.4.12 cần ≥ 4.13). Tạo lại `.venv/` (bản cũ trỏ tới
  Python không còn) theo đúng môi trường đã chạy; ghi phiên bản vào `env/requirements-runtime.txt`; `.python-version` → 3.12.
- `train_steer.py`: `--x_factor` mặc định 1,5 thay vì 1,0, khớp mọi lượt đã chạy.
- `analyze_boolq_margin.py`: bỏ khẳng định sai rằng token newline sau no/yes "triệt tiêu" (chỉ sửa chú thích và chuỗi mô tả
  trong `summary.json`; cách tính không đổi). `qwen2_base.py`: chú thích loss mask ghi đúng là có cả newline sau `<|im_end|>`.

### Kiểm chứng implementation (2026-10-13)

- Bản sửa fp32 có hiệu lực: sau khi load ở bf16 như lúc đánh giá, Full SFT fp32 đổi 76,1% tham số (bf16 cũ: 11,5%), kể cả
  35% weight có |w| ≥ 0,031; FAPM mới đổi 9,2%, dồn vào weight lớn đúng như công thức.
- Chạy `SFTTrainer` thật trên CPU với một model Qwen2 tí hon: loss mask chỉ phủ phần trả lời; `num_items_in_batch` bằng đúng
  tổng token assistant của batch tích luỹ; gradient tích luỹ của cả Steer lẫn CE mặc định (`chunked_nll`) khớp chính xác
  (sai khác 0,0) với gradient tính tay trên toàn batch.
- Không mẫu nào vượt 2.048 token (train dài nhất 1.137; prompt BoolQ dài nhất 1.270), nên không có gì bị cắt.
- 33 test CPU pass.

## 2026-10-12

- Sửa full fine-tune train thuần bf16 (xem `docs/experiments/05-bf16-diagnosis-fp32-rerun.md`): `train_sft.py`,
  `train_steer.py`, `train_l1.py` lưu weight fp32 theo mặc định; `fapm.py` tính ở fp32. Thêm `--optim`, `--max_steps`.
- Thêm `run_fp32_master_rerun.sh`, `eval/upload_models_wandb.py`, `benchmarks/stable_five/plot_fp32_rerun.py`.

## 2026-10-03 · Dọn code

Giữ source cần cho các thí nghiệm đã hoàn tất, và giữ L1 theo yêu cầu. Đã xoá 69 file:

| Đã xoá | Lý do |
|---|---|
| `methods/train_ewc.py` | Phương pháp không nằm trong các lượt đã chạy |
| `eval/gpu_margin_interference.py` | Chẩn đoán chưa chạy, được thay bằng probe gradient trên tham số thật |
| `eval/gpu_margin_comparison.py` | Runner chạy dở; các hàm dùng chung được tách nguyên văn sang `eval/gpu_probe_utils.py` |
| `legacy_eval/` (65 file) | Bộ đánh giá cũ, thay bằng `benchmarks/stable_five/run.py` |
| `tests/test_evaluation_parsers.py` | Test cho bộ đánh giá cũ |

Các thay đổi khác lúc đó: sửa khẳng định sai rằng newline sau no/yes của BoolQ triệt tiêu nhau; trỏ mặc định benchmark tới
các checkpoint hiện có; không đổi công thức loss, optimizer, dữ liệu hay kết quả đã lưu. Kiểm tra: 25 test CPU pass,
mọi file Python biên dịch được, `train_l1.py` trùng từng byte với bản trước khi dọn. Bản sao tạm lúc đó
(`/tmp/sdft-code-cleanup-20261003T182356Z`) không còn được đảm bảo tồn tại.

## Vấn đề đã biết, chưa sửa

- `train_l1.py`: phạt |θ| (kéo về 0) thay vì |θ − θ_base|; dùng `adamw_torch` nên với weight fp32 sẽ hết bộ nhớ trên 80 GB.
- Một số script phân tích viết cứng số liệu trong phần diễn giải (`analyze_margin_mechanism.py`,
  `audit_margin_artifacts.py`, `analyze_completed_margin_study.py`).
- Đường dẫn mặc định trong `benchmarks/stable_five/run.py` và các script phân tích còn trỏ tới checkpoint bf16 đã xoá.
- `uv.lock` / `pyproject.toml` lỗi thời so với môi trường thật (xem `docs/setup.md`).
