import argparse
import os
import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-7B")
    parser.add_argument("--peft_model", type=str, default="outputs/steer-qwen2-7b")
    parser.add_argument("--output_dir", type=str, default="outputs/steer-qwen2-7b-merged")
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

    print("Loading tokenizer configuration...")
    tokenizer_source = (
        args.peft_model
        if os.path.exists(os.path.join(args.peft_model, "tokenizer_config.json"))
        else args.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)

    merged_model.config.eos_token_id = tokenizer.eos_token_id
    merged_model.config.pad_token_id = tokenizer.pad_token_id
    if merged_model.generation_config is not None:
        merged_model.generation_config.eos_token_id = tokenizer.eos_token_id
        merged_model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"Saving merged model to {args.output_dir}...")
    merged_model.save_pretrained(args.output_dir)
    print("Saving tokenizer...")
    tokenizer.save_pretrained(args.output_dir)

    if args.push_to_hub and args.repo_name:
        print(f"Pushing to HuggingFace Hub: {args.repo_name}...")
        merged_model.push_to_hub(args.repo_name)
        tokenizer.push_to_hub(args.repo_name)
        print("Successfully pushed to Hub!")
    
    print("Done!")

if __name__ == "__main__":
    main()
