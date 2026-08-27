import argparse
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="Apply FAPM to merge SFT model back into Base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path or name of the pre-trained base model")
    parser.add_argument("--sft_model", type=str, required=True, help="Path to the Full SFT fine-tuned model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the merged FAPM model")
    parser.add_argument("--k_ratio", type=float, default=0.1, help="Ratio of parameters to keep (default: 0.1)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"[FAPM] Đang load Base Model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu", 
        trust_remote_code=True
    ).eval()
    
    print(f"[FAPM] Đang load SFT Model: {args.sft_model}")
    model_sft = AutoModelForCausalLM.from_pretrained(
        args.sft_model, 
        torch_dtype=torch.bfloat16, 
        device_map="cpu", 
        trust_remote_code=True
    ).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    
    weights = model.state_dict()
    weights_sft = model_sft.state_dict()
    
    print(f"[FAPM] Bắt đầu tính toán Task Vector và merge với tỷ lệ giữ lại k = {args.k_ratio}")
    for key, value in weights.items():
        if key not in weights_sft:
            continue
            
        # Không can thiệp vào embedding hoặc layernorm nếu cần, nhưng code gốc áp dụng cho mọi param.
        # Nếu RAM GPU ít, có thể chạy trên CPU nhưng chậm hơn. Ở đây ta chuyển tensor sang GPU để tính toán nhanh, sau đó chuyển lại.
        w_base = weights[key].cuda()
        w_sft = weights_sft[key].cuda()
        
        # Task vector
        tensor = w_sft - w_base
        
        k = int(tensor.numel() * args.k_ratio)
        if k == 0:
            continue
            
        lamda = 1.0 * w_base.abs().mean()
        
        # Tính điểm t
        # tensor.abs() / w_base.abs() có thể gặp chia cho 0, thêm epsilon.
        eps = 1e-8
        t = tensor.abs() - lamda * (tensor.abs() / (w_base.abs() + eps))
        
        # Lấy top k
        indices = torch.argsort(t.view(-1), descending=True)[:k]
        
        mask = torch.zeros_like(tensor)
        mask.view(-1)[indices] = 1
        tensor.mul_(mask)
        
        # Merge
        weights[key] = (w_base + tensor).cpu()
        
        # Clean up memory
        del w_base, w_sft, tensor, t, mask
        torch.cuda.empty_cache()

    model.load_state_dict(weights)
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if model.generation_config is not None:
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    print(f"[FAPM] Đang lưu mô hình mới tại: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("[FAPM] Xong!")

if __name__ == "__main__":
    main()
