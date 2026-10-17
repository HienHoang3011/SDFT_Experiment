# 04 · Hướng cập nhật từ checkpoint đã có

**Trạng thái:** hoàn tất 2026-10-04 (phần 2a: 7 giờ 8 phút). Artifact: `artifacts/samsum/update-directions-existing-20261003/`
(`ANALYSIS_VI.md` là báo cáo chính; `RUN_SETUP.md` là thiết lập thực tế).

> Thí nghiệm này dùng checkpoint Full SFT / Steer-Full / FAPM bf16 cũ, nên số liệu của ba model đó phản ánh lỗi bf16
> (xem [05](05-bf16-diagnosis-fp32-rerun.md)). Các checkpoint này đã bị xoá nên không chạy lại được trên chúng.

## Phần 1 · Độ dịch chuyển cuối cùng

Source: `eval/analyze_checkpoint_directions.py`. Tính ΔW = W_final − W_Base trên toàn bộ 7.615.616.512 tham số (LoRA dùng
weight hiệu dụng alpha/r · B@A), báo cosine, chuẩn, độ phủ và lượng thay đổi FAPM giữ lại. Kết quả: `part1/REPORT.md`,
`part1/directions.json`.

## Phần 2a · Hướng cập nhật cục bộ

Source: `eval/probe_checkpoint_updates.py`, `eval/finish_checkpoint_direction_study.py`, module `eval/update_direction_metrics.py`.

- Ở mỗi checkpoint đã train (Full SFT, LoRA, Steer-Full, Steer-LoRA), thử một bước CE và một bước Steer trên cùng batch
  64 mẫu, cùng seed dropout; khôi phục checkpoint trước mỗi lần thử. AdamW8bit mới mỗi lần (LR 1e-5), không có moment cũ.
- Đo cosine gradient CE–Steer, cosine bước AdamW thực tế, chuẩn, từng layer; probe 64 mẫu SAMSum dev và 512 câu BoolQ.
- 4 batch calibration rồi tự mở rộng đều theo ngân sách 7–8 giờ: 49 batch/checkpoint, 196 cặp, 392 lần thử.
- Kết quả: `part2a/BATCH_ANALYSIS.md`, `part2a/results.json`, `direction_summary.png`; kiểm tra: `RESULT_VALIDATION.json`.

## Giới hạn

Moment optimizer mới không khôi phục hướng của các bước train cũ; CI theo batch chẩn đoán trên tập probe cố định và một
training seed. Hướng khác nhau không tự chứng minh gây quên; probe 64 SAMSum + 512 BoolQ không thay benchmark đầy đủ.

## Chạy lại

Script shell gốc `run_checkpoint_direction_study.sh` đã bị xoá. Lệnh tương đương:

```bash
PY=/root/sdft-fp32-runs/venv/bin/python
$PY -u eval/analyze_checkpoint_directions.py --artifact-root <run> --output-dir <out>/part1
$PY -u eval/probe_checkpoint_updates.py --artifact-root <run> --output-dir <out>/part2a --batches 4 --accumulation 64
$PY -u eval/finish_checkpoint_direction_study.py --run-root <out> --target-hours 7.5
```

Các script này mặc định trỏ tới checkpoint bf16 cũ đã xoá; cần truyền `--artifact-root` mới.
