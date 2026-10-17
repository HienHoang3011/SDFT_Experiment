# Các phương pháp

| Phương pháp | Source | Mô tả |
|---|---|---|
| Base | — | Qwen2-7B, không train |
| Full SFT | `methods/train_sft.py` | Cross-entropy, cập nhật toàn bộ weight |
| LoRA | `methods/train_sft_lora.py` | Cross-entropy, chỉ cập nhật adapter |
| Steer-LoRA / Steer-Full | `methods/train_steer.py` (`--x_factor 1.5`, thêm `--disable_peft` cho Full) | Self-margin KL, công thức bên dưới |
| Steer + neo Base | `methods/train_steer.py --base_kl_weight 1.0`, `methods/base_anchored_steer.py` | Steer + KL(Base ‖ model); chỉ hỗ trợ LoRA |
| FAPM | `methods/fapm.py` | Hậu xử lý Full SFT, không train riêng |
| L1 | `methods/train_l1.py` | Giữ lại theo yêu cầu, chưa chạy; xem ghi chú cuối trang |

## Steer

Ký hiệu: z là logit, p = softmax(z), y là token đúng, z_max là logit lớn nhất trong các token khác y.

```
m      = log(1 + x) = log 2,5 ≈ 0,916            (x = 1,5)
δ      = max(0, z_max + m − z(y))
z̃(y)   = z(y) + δ,   các logit khác giữ nguyên
q      = softmax(z̃)                              (detach, không lấy gradient)
Loss   = trung bình theo token của KL(q ‖ p)
```

Target lấy từ chính model hiện tại; đạt margin (p(y) ≥ 2,5 × xác suất đối thủ mạnh nhất) thì δ = 0 và loss bằng 0.
Không có số hạng nào neo về Base. `train_steer.py` mặc định x = 1,5 (trước 2026-10-13 mặc định là 1,0; mọi lượt
chạy đều truyền rõ 1,5).

Loss mask (cả CE lẫn Steer) phủ nội dung trả lời, `<|im_end|>` và ký tự xuống dòng ngay sau nó.

**Đẩy logit lên δ thay đổi softmax thế nào.** Với Z = 1 + p(y)·(eᵟ − 1): q(y) = p(y)·eᵟ/Z, q(j) = p(j)/Z với j ≠ y,
nên các token khác giữ nguyên tỉ lệ với nhau. KL(q ‖ p) = q(y)·δ − log Z. Với δ của Steer, Z = 1 + 2,5·p_max − p(y).

**Steer là cross-entropy có trọng số theo token.** Vì q được detach, gradient theo logit là p − q = w·(p − one-hot(y)) với
w = 1 − 1/Z ∈ [0, 1). Trọng số bằng 0 khi đã đạt margin, khoảng 0,69 khi có một đối thủ sai rất tự tin (p_max = 0,9),
khoảng 0,2 khi phân phối phẳng (p_max = 0,1). CE luôn có w = 1. Công thức này đã được kiểm chứng số trên chính lớp
`SteeredLoss` (sai số gradient < 2·10⁻⁶, `eval/audit_margin_artifacts.py`).

## FAPM

```
Δ       = W_SFT − W_base
λ       = trung bình |W_base| trong từng tensor
t       = |Δ| · (1 − λ / |W_base|)
mask    = 1 ở k = 10% phần tử có t lớn nhất trong tensor
W_FAPM  = W_base + Δ ⊙ mask
```

`methods/fapm.py` khớp code tác giả (secretflow/ACoLab, `PaperCode/FAPM/FAPM.py`, bài EMNLP 2025), chỉ thêm ε = 1e-8
ở mẫu số. Từ 2026-10-12, Δ và t được tính ở fp32, chỉ kết quả merge ép về bf16.

Đặc điểm cần biết:
- t > 0 chỉ khi |W_base| > λ: ưu tiên thay đổi trên weight lớn hơn trung bình.
- Phần tử không đổi (Δ = 0) có t = 0, xếp trên mọi phần tử t âm. Khi phần lớn thay đổi nằm trên weight nhỏ,
  top-k bị lấp bằng số 0 (đây là điều đã xảy ra với checkpoint bf16 cũ).
- Điểm t không dùng gradient hay loss.

## L1

`methods/train_l1.py`: Loss = CE + λ·Σ|θ| trên mọi tham số train được (mặc định λ = 1e-8, `adamw_torch`,
lưu checkpoint mỗi 100 bước). Hai khác biệt so với baseline cần sửa nếu muốn so sánh có kiểm soát:
phạt |θ| kéo weight về 0 chứ không về Base (|θ − θ_base|), và optimizer khác. Bản này tạo bản fp32 của
toàn bộ 7B tham số ở mỗi micro-batch nên dễ hết bộ nhớ.
