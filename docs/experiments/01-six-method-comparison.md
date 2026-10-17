# 01 · So sánh sáu phương pháp, 2 epoch

**Trạng thái:** hoàn tất (2026-09-26 → 09-28). Artifact: `artifacts/samsum/fair-methods-2epoch-log25-tmux/`.

Base, Full SFT, LoRA, Steer-LoRA, Steer-Full, FAPM; thiết lập chung ở [setup.md](../setup.md), công thức ở [methods.md](../methods.md).

> **Lưu ý.** Full SFT, Steer-Full và FAPM của lượt này train với weight bf16 thuần nên phần lớn weight không đổi
> (xem [05](05-bf16-diagnosis-fp32-rerun.md)). Ba checkpoint này đã bị xoá ngày 2026-10-13; điểm số giữ lại để đối chiếu.
> LoRA, Steer-LoRA và Base không bị ảnh hưởng (adapter fp32) và vẫn là kết quả hiện hành.

## Kết quả (%)

| Benchmark | Base | Full SFT (bf16) | LoRA | Steer-LoRA | Steer-Full (bf16) | FAPM (bf16) |
|---|---:|---:|---:|---:|---:|---:|
| SAMSum ROUGE-L | 32,93 | 46,04 | 46,24 | 44,23 | 43,69 | 32,93 |
| GSM8K (1.024 token) | 69,67 | 47,46 | 61,18 | 63,53 | 63,31 | 69,83 |
| HumanEval | 72,56 | 62,80 | 64,02 | 61,59 | 64,63 | 72,56 |
| IFEval | 34,94 | 30,50 | 28,65 | 22,74 | 23,11 | 33,46 |
| BoolQ | 80,55 | 77,52 | 83,18 | 62,20 | 62,17 | 80,73 |
| BBH | 48,07 | 43,62 | 44,77 | 42,73 | 32,78 | 48,59 |

`stable-five/` chạy GSM8K với độ dài sinh mặc định, làm lời giải dài của Base bị cắt (Base 62,55). Dòng GSM8K ở trên lấy từ
`stable-five-gsm8k-1024/`, chạy lại cho cả sáu model với 1.024 token.

## Nhận xét chính

- **Steer so với CE thường (cặp LoRA, cặp sạch):** thua chắc ở BoolQ ChatML (−21,0 điểm, CI [−22,8; −19,2]), IFEval (−5,9),
  ROUGE-L (−2,0) và dev NLL (+0,11). GSM8K +2,4 nhưng CI chứa 0.
- **BoolQ:** với prompt ChatML, Steer trả lời "yes" cho 100% câu (62,2% = tỉ lệ nhãn yes); với prompt raw vẫn ngang Base
  (85,6% so với 84,8%, AUC 0,927). Kiến thức còn, cách đọc đáp án trong ChatML bị lệch.
- **Phân phối:** entropy trên SAMSum dev của Steer tăng (1,39 so với Base 0,90, LoRA 0,81); xác suất token đúng giảm mạnh ở
  token Base đã chắc. Phù hợp với w ≈ 0 ở token phẳng và token đã đạt margin.
- **Định dạng:** mọi model fine-tune hay mở đầu câu trả lời bằng thẻ HTML (`<ol>`, `<code>`, `<p>`…), học từ format `<answer>`;
  Steer-LoRA nhiều nhất trên IFEval (51%). BBH của Steer-Full có 18% câu trả lời rỗng.

## Phân tích đi kèm (trong `reports/`)

| Thư mục | Nội dung | Script |
|---|---|---|
| `margin-mechanism/` | Phân phối SAMSum dev đầy đủ, BoolQ raw/ChatML, cơ chế margin | `eval/eval_kl.py`, `eval/analyze_boolq_margin.py`, `eval/analyze_margin_mechanism.py` |
| `margin-audit-20261003T123623Z/` | Tính lại điểm từ artifact, paired bootstrap, kiểm chứng gradient | `eval/audit_margin_artifacts.py` |
| `gpu-actual-gradients-20261003-run1/` | Gradient CE/Steer trên tham số thật (2 ví dụ train, chỉ là probe) | `eval/gpu_actual_parameter_gradients.py` |
| `kl/`, `distribution-comparison*/` | So sánh phân phối với các cách chọn mẫu khác nhau | `eval/eval_kl.py`, `eval/plot_distribution_comparison.py` |
| `<model>/downstream_results.json` | ROUGE-L và câu tóm tắt sinh ra | `eval/eval_samsum.py` |

`reports/gpu-margin-comparison-20261003-run1/` là lượt chạy dở, không phải kết quả.

## Chạy lại

Các script shell gốc (`run_samsum_*.sh`) đã bị xoá. Lệnh tương đương, chạy từ thư mục project:

```bash
PY=/root/sdft-fp32-runs/venv/bin/python
export PYTHONPATH=methods
COMMON="--model_name_or_path artifacts/base/Qwen2-7B --num_train_epochs 2 --learning_rate 1e-5 --seed 42"
$PY methods/train_sft_lora.py $COMMON --output_dir <out>/lora
$PY methods/train_steer.py    $COMMON --x_factor 1.5 --output_dir <out>/steer_lora
$PY eval/eval_samsum.py --model_path <out>/lora --output_dir <reports>/lora
$PY benchmarks/stable_five/run.py --output-dir <out>/stable-five --model-path lora=<out>/lora --models lora
```

Full SFT, Steer-Full và FAPM: dùng `run_fp32_master_rerun.sh` (xem [05](05-bf16-diagnosis-fp32-rerun.md)).
