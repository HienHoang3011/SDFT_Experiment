# Bố cục `artifacts/`

`artifacts/` nằm ngoài git (xem `.gitignore`). Checkpoint lớn của lượt fp32 nằm ở `/root/sdft-fp32-runs/models/`.

| Đường dẫn | Nội dung | Tài liệu |
|---|---|---|
| `base/Qwen2-7B/` | Checkpoint Base | — |
| `samsum/fair-methods-2epoch-log25-tmux/` | So sánh sáu phương pháp, 2 epoch | [01](experiments/01-six-method-comparison.md) |
| `samsum/fixed-margin-causal-128step-20261003/` | Sáu đối chứng 128 bước | [02](experiments/02-fixed-margin-controls.md) |
| `samsum/update-directions-existing-20261003/` | Hướng cập nhật từ checkpoint | [04](experiments/04-update-directions.md) |
| `samsum/fp32-master-2epoch-20261012/` | Lượt chạy lại fp32 | [05](experiments/05-bf16-diagnosis-fp32-rerun.md) |
| `backups/artifacts_images_markdown_json_20261004_042904.zip` | Bản sao ảnh/md/json của `artifacts/` và tài liệu ngày 2026-10-04 | — |

Thư mục đã xoá ngày 2026-10-13: `samsum/steer-base-anchor-2epoch-lambda1-20261003/` (thí nghiệm [03](experiments/03-steer-base-anchor.md)),
`samsum/tensor-change-analysis-20261012/` (phân tích trong [05](experiments/05-bf16-diagnosis-fp32-rerun.md)).

## `fair-methods-2epoch-log25-tmux/`

| Đường dẫn | Nội dung |
|---|---|
| `REPORT.md` | Báo cáo tự sinh (`eval/generate_report.py`) |
| `models/lora`, `models/steer_lora` | Adapter LoRA. Ba checkpoint bf16 `full_sft`, `steer_full`, `fapm` đã xoá |
| `reports/fairness_manifest.json` | Cấu hình chung của lượt chạy |
| `reports/<model>/` | ROUGE-L, câu sinh ra |
| `reports/margin-mechanism/` | Phân phối SAMSum dev, BoolQ raw/ChatML, `boolq.validation.jsonl` dùng chung |
| `reports/margin-audit-*/`, `reports/gpu-actual-gradients-*/`, `reports/kl/`, `reports/distribution-comparison*/` | Các phân tích đi kèm |
| `stable-five/`, `stable-five-gsm8k-1024/` | Điểm lm-eval, từng mẫu ở `models/<model>/` |
| `logs/run.log`, `logs/eval_resume.log` | Log train và đánh giá |

## `fixed-margin-causal-128step-20261003/`

`ANALYSIS.md` (báo cáo chính), `REPORT.md` (tự sinh), `EXPERIMENT_DESIGN.md`, `run_manifest.json`, `trajectories.json/png`,
`<arm>.train.jsonl` (log từng bước), `<arm>.step<N>.boolq-{raw,chat}.json` (điểm BoolQ theo bước), `<arm>.rouge.json`,
`paired_contrasts.json`, `analysis_statistics.json`, `models/<arm>/` (adapter), `logs/`.

## `update-directions-existing-20261003/`

`ANALYSIS_VI.md` (báo cáo chính), `RUN_SETUP.md`, `part1/` (độ dịch chuyển cuối), `part2a/` (thử từng bước, `results.json`,
`BATCH_ANALYSIS.*`), `direction_summary.png/pdf`, `RESULT_VALIDATION.json`, file trạng thái `RUN_STATUS.json`, `BUDGET_*.json`, `logs/`.

## `fp32-master-2epoch-20261012/`

| Đường dẫn | Nội dung |
|---|---|
| `reports/fairness_manifest.json`, `reports/precision_manifest.json` | Cấu hình, độ chính xác, optimizer |
| `reports/<model>/` | ROUGE-L của Full SFT, Steer-Full, FAPM |
| `reports/margin-mechanism/` | Phân phối SAMSum dev và BoolQ raw/ChatML (kèm LoRA, Steer-LoRA để so cùng lượt) |
| `stable-five/` | Điểm 5 benchmark (đang chạy) |
| `logs/` | Log từng bước, `phases.log`, `environment.txt` |
| `plot_training_curves.py`, `training_curves_*.png`, `scores-*.png` | Hình |
