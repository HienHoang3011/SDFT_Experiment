import argparse
import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on science test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--max_new_tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save evaluation results (defaults to model_path)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for generation")
    return parser.parse_args()

def load_model_and_tokenizer(model_path):
    """Load model using transformers for Windows compatibility."""
    print(f"Loading model from {model_path}")
    
    # Check if it's a PEFT model by looking for adapter_config.json
    is_peft = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    if is_peft:
        print("Detected PEFT adapter. Loading base model first, then applying adapter...")
        with open(os.path.join(model_path, "adapter_config.json"), "r") as f:
            config = json.load(f)
        base_model_path = config.get("base_model_name_or_path")
        
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, padding_side='left')
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            if "<|endoftext|>" in tokenizer.vocab:
                tokenizer.pad_token = "<|endoftext|>"
            else:
                tokenizer.pad_token = tokenizer.eos_token
            
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        
        # Đảm bảo có pad_token
        if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
            if "<|endoftext|>" in tokenizer.vocab:
                tokenizer.pad_token = "<|endoftext|>"
            else:
                tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
    return model, tokenizer

def load_test_data():
    """Load science test dataset."""
    path = 'data/science_data/eval_data'
    print(f"Loading science test dataset from {path}")
    data = Dataset.load_from_disk(path)
    return data

def generate_responses(model, tokenizer, prompts, batch_size=4, max_new_tokens=2048, temperature=0.0):
    """Generate responses using standard HF pipeline."""
    formatted_prompts = []
    for prompt in prompts:
        formatted_prompt = tokenizer.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted_prompt)

    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    all_outputs = []
    
    # Sinh text theo batch
    for i in range(0, len(formatted_prompts), batch_size):
        batch_prompts = formatted_prompts[i:i+batch_size]
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
        
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            
        # Tách phần prompt ra khỏi output để chỉ lấy câu trả lời mới sinh ra
        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[:, input_length:]
        
        batch_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        all_outputs.extend(batch_texts)
        
        if (i + batch_size) % 20 == 0:
            print(f"  Processed {min(i+batch_size, len(formatted_prompts))}/{len(formatted_prompts)}")
            
    return all_outputs

def extract_xml_answer(text: str) -> str:
    """Extract answer from XML-formatted text."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def evaluate_correctness(responses, answers):
    """
    Evaluate if responses match the golden answers.
    Returns list of scores (1 for correct, 0 for incorrect).
    """
    results = []
    for response, answer in zip(responses, answers):
        extracted = extract_xml_answer(response)
        results.append(1 if extracted == answer else 0)
    return results


def main():
    args = parse_args()

    # Load model and data
    llm, tokenizer = load_model_and_tokenizer(args.model_path)
    test_data = load_test_data()

    prompts = [example['prompt'] for example in test_data]
    answers = [example['answer'] for example in test_data]

    # Generate responses
    responses = generate_responses(
        model=llm, 
        tokenizer=tokenizer, 
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # Evaluate correctness
    print("\nEvaluating responses...")
    scores = evaluate_correctness(responses, answers)
    accuracy = np.mean(scores)

    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 60)

    # Save results
    output_dir = args.output_dir if args.output_dir else args.model_path
    os.makedirs(output_dir, exist_ok=True)

    results_to_save = {
        "accuracy": float(accuracy),
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "per_sample_scores": scores,
        "config": {
            "model_path": args.model_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
        }
    }

    output_path = os.path.join(output_dir, "eval_results.json")
    with open(output_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\nSaved results to {output_path}")

    # Save responses for inspection
    responses_path = os.path.join(output_dir, "eval_responses.json")
    with open(responses_path, "w") as f:
        json.dump([
            {
                "prompt": prompts[i],
                "response": responses[i],
                "answer": answers[i],
                "correct": bool(scores[i])
            }
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved responses to {responses_path}")


if __name__ == "__main__":
    main()