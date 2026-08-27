import argparse
import os
import json
import torch
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on MedQA test set")
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
    """Load MedQA test dataset."""
    path = 'data/medqa_data/eval_data'
    print(f"Loading MedQA test dataset from {path}")
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
    if "<answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def evaluate_correctness(responses, answers, prompts=None):
    """
    Evaluate if responses match the golden answers using XML tags or heuristics.
    Returns list of scores (1 for correct, 0 for incorrect).
    """
    results = []
    for i, (full_response, answer) in enumerate(zip(responses, answers)):
        if "<answer>" in full_response:
            eval_text = extract_xml_answer(full_response)
        else:
            eval_text = full_response
            
        is_correct = 0
        prompt_text = ""
        if prompts and i < len(prompts):
            p = prompts[i]
            if isinstance(p, list) and len(p) > 0 and 'content' in p[0]:
                prompt_text = p[0]['content']
            elif isinstance(p, str):
                prompt_text = p
                
        correct_letter = None
        all_options = []
        for line in prompt_text.split('\n'):
            line = line.strip()
            if re.match(r'^[A-E]\.', line):
                opt_text = line[2:].strip()
                all_options.append(opt_text)
                if answer.lower() in opt_text.lower() or opt_text.lower() in answer.lower():
                    correct_letter = line[0].upper()
                    
        clean_eval = eval_text.strip().strip('.').lower()
        clean_answer = answer.strip().strip('.').lower()
        
        # Fair direct comparison
        if clean_eval == clean_answer:
            results.append(1)
            continue
            
        if correct_letter:
            if clean_eval == correct_letter.lower():
                results.append(1)
                continue
                
            match = re.search(r'(?i)answer is\s*:?\s*([A-E])\b', eval_text)
            if match:
                if match.group(1).upper() == correct_letter:
                    is_correct = 1
            else:
                appearances = [opt for opt in all_options if opt.lower() in eval_text.lower()]
                if len(appearances) == 1 and appearances[0].lower() == answer.lower():
                    is_correct = 1
                elif re.search(rf'\b{correct_letter}\.\s', eval_text) or re.search(rf'^[A-E]\.\s*{re.escape(answer)}', eval_text, re.IGNORECASE | re.MULTILINE):
                    is_correct = 1
        else:
            if answer.lower() in eval_text.lower() and len(eval_text) < len(answer) + 50:
                is_correct = 1
                
        results.append(is_correct)
    return results


def main():
    args = parse_args()

    # Load model and data
    llm, tokenizer = load_model_and_tokenizer(args.model_path)
    test_data = load_test_data()

    prompts = [example['messages'] for example in test_data]
    # In format_sample we made messages user/assistant, but during eval we only want to pass the user prompt.
    # The previous code used example['prompt'] but passed it to apply_chat_template. Wait!
    # apply_chat_template expects a list of dicts. If we pass the string example['prompt'] it won't work well unless formatted.
    # In eval_science.py, prompts = [example['prompt'] for example in test_data]. Wait, if example['prompt'] was a string, how did apply_chat_template work? It expects a list of messages. Let's fix it by passing just the user message.
    eval_prompts = [[{"role": "user", "content": example['prompt']}] for example in test_data]
    answers = [example['answer'] for example in test_data]

    # Generate responses
    responses = generate_responses(
        model=llm, 
        tokenizer=tokenizer, 
        prompts=eval_prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature
    )

    # Evaluate correctness
    print("\nEvaluating responses...")
    scores = evaluate_correctness(responses, answers, eval_prompts)
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
                "prompt": eval_prompts[i][0]['content'],
                "response": responses[i],
                "answer": answers[i],
                "correct": bool(scores[i])
            }
            for i in range(len(responses))
        ], f, indent=2)
    print(f"Saved responses to {responses_path}")


if __name__ == "__main__":
    main()
