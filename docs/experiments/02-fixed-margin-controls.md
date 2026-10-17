# 02 · Sáu đối chứng cùng ngân sách, margin cố định (128 bước)

**Trạng thái:** hoàn tất 2026-10-03 (khoảng 4 giờ 10 phút GPU). Artifact: `artifacts/samsum/fixed-margin-causal-128step-20261003/`
(`ANALYSIS.md` là báo cáo chính; `EXPERIMENT_DESIGN.md` là bản thiết kế lúc chạy).

Source: `methods/train_fixed_margin_controls.py`; phân tích: `eval/analyze_completed_margin_study.py`.

## Thiết kế

| Arm | Loss | Câu hỏi |
|---|---|---|
| `ce` | CE | Baseline LoRA thường |
| `steer` | Steer gốc | Tái hiện khoảng cách với CE |
| `ce_half` | 0,5 × CE | Chỉ giảm độ mạnh gradient có gây lỗi không? |
| `steer_floor` | Trọng số token w → max(w; 0,1) | Việc ngừng đẩy token đã đạt margin có góp phần không? |
| `ce_anchor` | CE + KL(Base ‖ model) | Neo Base ảnh hưởng baseline thế nào? |
| `steer_anchor` | Steer + KL(Base ‖ model) | Neo Base có chặn được lỗi của Steer không? |

- Cùng Base, cùng khởi tạo LoRA, cùng thứ tự dữ liệu, seed 42. 128 bước × 64 = 8.192 mẫu train mỗi arm.
- LoRA 256/512/0,3; AdamW8bit, LR đỉnh 1e-5; dùng 128 bước đầu của lịch cosine 458 bước (warmup 14).
- Margin cố định log 2,5; floor 0,1; hệ số KL 1. Teacher Base: tắt adapter, eval mode.
- Probe ở bước 0, 32, 64, 96, 128: 64 mẫu SAMSum dev (tối đa 32 vị trí/mẫu), 512 câu BoolQ raw và ChatML.
- Cuối train: ROUGE-L trên 128 mẫu SAMSum test.

## Kết quả ở bước 128

| Arm | ROUGE-L % (128 mẫu) | BoolQ ChatML % | Đoán yes % | AUC | BoolQ raw % | ΔP(y), token Base ≥ 0,9 |
|---|---:|---:|---:|---:|---:|---:|
| ce | 47,84 | 83,79 | 67,77 | 0,917 | 86,52 | −0,031 |
| steer | 44,29 | 62,11 | 100,00 | 0,851 | 85,74 | −0,168 |
| ce_half | 47,75 | 82,62 | 70,51 | 0,915 | 85,94 | −0,031 |
| steer_floor | 45,00 | 62,11 | 100,00 | 0,793 | 86,13 | −0,117 |
| ce_anchor | 43,59 | 82,42 | 69,92 | 0,908 | 85,55 | −0,016 |
| steer_anchor | 37,75 | 81,64 | 70,70 | 0,908 | 85,16 | −0,025 |

Kết luận:
- `ce_half` gần như trùng `ce`: gradient yếu không phải nguyên nhân (AdamW chuẩn hoá độ lớn gradient).
- `steer_floor` không cứu được BoolQ: vẫn đoán yes 100%.
- Chỉ `steer_anchor` cứu được BoolQ (+19,5 điểm so với `steer`), nhưng ROUGE-L mất thêm 6,5 điểm.
- Steer đoán yes 100% ngay từ bước 32, trong khi CE giữ khoảng 70%.

Điểm trên tập con 128 bước không thay cho benchmark đầy đủ của thí nghiệm 01. Chỉ một seed.

## Chạy lại

Script shell gốc `run_fixed_margin_causal_study.sh` đã bị xoá. Lệnh tương đương:

```bash
PY=/root/sdft-fp32-runs/venv/bin/python
$PY -u methods/train_fixed_margin_controls.py --output-dir artifacts/samsum/<new-controls> \
  --steps 128 --accumulation 64 --probe-every 32 --dev-samples 64 --boolq-samples 512 --rouge-samples 128
$PY eval/analyze_completed_margin_study.py --root artifacts/samsum/<new-controls>
```

Script cần `boolq.validation.jsonl` và `stable-five/summary.json` của thí nghiệm 01.
