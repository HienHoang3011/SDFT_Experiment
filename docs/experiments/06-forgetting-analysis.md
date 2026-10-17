# 06 · Vì sao model quên: tổng hợp phân tích lượt fp32

**Trạng thái:** 2026-10-14. Dùng sáu model của [05](05-bf16-diagnosis-fp32-rerun.md), chấm trong cùng một môi trường.
Chi tiết và số liệu gốc ở `artifacts/samsum/fp32-master-2epoch-20261012/`:
`benchmark-analysis/` (từng mẫu), `fapm-analysis/` (biến thể task vector), `forgetting-mechanism/` (weight, logit, gradient).

## Điểm

| | Base | LoRA | Steer-LoRA | Full SFT | Steer-Full | FAPM |
|---|---:|---:|---:|---:|---:|---:|
| SAMSum ROUGE-L | 33,06 | **46,27** | 44,06 | 44,89 | 43,73 | 41,27 |
| GSM8K | 70,74 | 60,96 | 63,23 | 38,89 | 32,90 | **66,03** |
| HumanEval | 70,12 | 65,24 | 61,59 | 59,76 | 58,54 | **67,07** |
| IFEval | 32,72 | **28,10** | 23,11 | 17,56 | 13,31 | 25,51 |
| BoolQ | 80,64 | **83,06** | 62,20 | 57,43 | 62,17 | 63,58 |
| BBH | 48,55 | **44,66** | 42,80 | 33,79 | 33,56 | 39,27 |

## Kết luận chính

1. **Phần lớn forgetting là lan định dạng, không phải mất kiến thức.**
   - Full SFT và Steer-Full trả lời 95–100% câu GSM8K/BBH trong `<answer>…</answer>`, ngắn và bỏ suy luận.
   - KL(Base ‖ model) trên lời giải của Base: 14,7 nat ở token đầu, 0,23 ở các token sau.
   - Mồi "Let's solve this step by step." đưa GSM8K của Full SFT từ 40,4 lên 57,2. BoolQ với prompt thô: mọi model 85–86%.
2. **Định dạng được quyết định ở các lớp muộn.**
   - Logit lens: P(mở thẻ) của Full SFT từ 0,01 ở lớp 22 lên 0,99 ở lớp 25.
   - Chỉ giữ update của lớp 21–27 + head: 20/20 câu mở `<answer>`.
   - Chỉ giữ update của lớp 0–20: ROUGE-L 44,9 (bằng Full SFT), không mở thẻ, nhưng không dừng được.
   - Bỏ update của lớp 21–27 (giữ phần còn lại): ROUGE-L 45,1, gần như không bọc thẻ, nhưng chỉ 56% câu GSM8K dừng đúng,
     GSM8K 44,0. Đóng băng lớp muộn không phải cách chữa đơn giản.
3. **Gradient SAMSum gần vuông góc gradient task chung** (cos −0,006). Xung đột chỉ mạnh ở các vector norm (norm cuối −0,90).
   Forgetting đến từ việc task vector quá lớn và vượt ngưỡng định dạng, không đến từ xung đột trực tiếp ở đa số tham số.
4. **Thu nhỏ task vector là cách giảm quên hiệu quả nhất, bất kể cách thu nhỏ.**
   - α·Δ với α = 0,5: ROUGE-L 45,3, GSM8K 66,8 (Full SFT: 44,8 / 40,4).
   - GSM8K giảm 20 điểm chỉ trong đoạn α từ 0,75 lên 1, tức dạng ngưỡng.
   - FAPM, top-10% theo |Δ| và a·Δ cùng năng lượng đều cho GSM8K 66–70, cả trên Full SFT lẫn Steer-Full.
5. **FAPM tốt chủ yếu nhờ thu nhỏ Δ, không nhờ tiêu chí t = |Δ|(1 − λ/|W|).** Mask theo phần tử còn gây một lỗi:
   FAPM gõ token kết thúc `<|im_end|>` thành chữ thường (lm-eval cắt tại đó, nên rỗng 17% IFEval, 8% BBH).
   Nguyên nhân không phải hàng token đặc biệt (trả lại update cho các hàng đó không đổi gì) mà là update giữa các lớp
   không còn khớp: chỉ giữ Δ của một khối lớp cũng gây lỗi này (11–32%), còn α·Δ thì không.
6. **Steer quên nhiều hơn CE ở cùng độ lớn update.**
   - cos(Δ, −g_task chung): Steer-Full −0,018 so với Full SFT −0,006; Steer-LoRA −0,011 so với LoRA −0,001.
   - Ở prompt ChatML, cả ba model Steer/FAPM luôn đoán "yes" ở BoolQ.
7. **LoRA giữ định dạng của Base** (thẻ ≤ 6%), update nhỏ (0,3% ‖W‖), hạng thấp. LoRA tốt nhất ở 4/6 chỉ số và còn tăng
   BoolQ nhờ bớt thiên về "yes".

## Giới hạn

- Một seed. Các probe GSM8K dùng 250 câu (sai số khoảng ±3 điểm).
- Mọi benchmark dùng ChatML cho cả Base; điểm tuyệt đối phụ thuộc định dạng (đặc biệt BoolQ).
- Gradient đo tại Base trên 64 ví dụ; xấp xỉ bậc 1 sai về độ lớn, chỉ dùng cosine.
