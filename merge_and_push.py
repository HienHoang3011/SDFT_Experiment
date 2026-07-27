import argparse
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--peft_model", type=str, default="outputs/sdft-qwen2.5-7b/checkpoint-7620")
    parser.add_argument("--output_dir", type=str, default="outputs/sdft-qwen2.5-7b-merged")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--repo_name", type=str, default="")
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading PEFT model from: {args.peft_model}...")
    model = PeftModel.from_pretrained(base_model, args.peft_model)

    print("Merging weights...")
    merged_model = model.merge_and_unload()

    print(f"Saving merged model to {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.repo_name:
        print(f"Pushing to HuggingFace Hub: {args.repo_name}...")
        merged_model.push_to_hub(args.repo_name)
        tokenizer.push_to_hub(args.repo_name)
        print("Successfully pushed to Hub!")
    
    print("Done!")

if __name__ == "__main__":
    main()
