# 03 · Steer-LoRA có neo Base, 2 epoch

**Trạng thái:** hoàn tất 2026-10-03. **Artifact đã bị xoá ngày 2026-10-13**: thư mục
`artifacts/samsum/steer-base-anchor-2epoch-lambda1-20261003/` không còn, kể cả weight adapter. Báo cáo và các file
json/md còn trong `artifacts/backups/artifacts_images_markdown_json_20261004_042904.zip`.

Source: `methods/train_steer.py --base_kl_weight 1.0`, `methods/base_anchored_steer.py`; báo cáo: `eval/report_anchored_control.py`.

## Thiết kế

- Giống Steer-LoRA của thí nghiệm 01, thêm `--x_factor 1.5 --base_kl_weight 1.0`.
- Loss: Steer + 1 × KL(Base đóng băng ‖ model). Teacher là Base với adapter tắt, eval mode; student vẫn có dropout.
- Sau train: phân phối trên toàn bộ SAMSum dev, BoolQ raw/ChatML (3.270 câu), ROUGE-L trên 819 mẫu test.

## Kết quả

| | Steer-LoRA | Steer-LoRA + neo Base |
|---|---:|---:|
| BoolQ ChatML | 62,2% | 81,5% |
| SAMSum ROUGE-L | 44,2% | 36,8% |

Neo về Base ngăn được việc BoolQ luôn đoán yes, nhưng làm mất khá nhiều chất lượng tóm tắt. λ = 1 được chọn trước,
không dò; đây là bằng chứng bổ sung, không phải một arm cùng ngân sách với thí nghiệm 02.

## Chạy lại

```bash
PY=/root/sdft-fp32-runs/venv/bin/python
PYTHONPATH=methods $PY methods/train_steer.py --model_name_or_path artifacts/base/Qwen2-7B \
  --x_factor 1.5 --base_kl_weight 1.0 --num_train_epochs 2 --learning_rate 1e-5 --seed 42 \
  --output_dir <out>/steer_lora_base_anchor
```
