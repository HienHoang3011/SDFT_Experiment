import os
import json
from datasets import load_dataset, Dataset

def prepare_medqa():
    print("Đang tải dữ liệu MedQA từ HuggingFace (GBaker/MedQA-USMLE-4-options)...")
    dataset = load_dataset('GBaker/MedQA-USMLE-4-options')
    
    # Hàm format một sample
    def format_sample(example):
        question = example['question']
        options = example.get('options', {})
        answer_text = example.get('answer', '')
        
        # Nếu options là chuỗi dict thì convert
        if isinstance(options, str):
            try:
                options = json.loads(options.replace("'", "\""))
            except:
                pass
                
        prompt = f"Question: {question}\n\nOptions:\n"
        if isinstance(options, dict):
            for key, val in options.items():
                prompt += f"{key}. {val}\n"
        else:
             prompt += str(options) + "\n"
             
        prompt += "\nPlease provide the correct answer."
        
        # Format đầu ra giống dataset cũ: output_text hoặc answer
        # Định dạng huấn luyện mong đợi: user - assistant
        # Script evaluate_science của bạn mong đợi <answer> ... </answer>
        output_text = f"<answer>{answer_text}</answer>"
        
        return {
            "prompt": prompt,
            "answer": answer_text,
            "output_text": output_text,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": output_text}
            ]
        }
    
    # Tập dữ liệu GBaker/MedQA-USMLE-4-options chỉ có 'train' và 'test'. 
    # Do đó, ta sẽ cắt 10% từ tập train ra làm tập dev (validation) để quá trình SFT đánh giá.
    print("Đang tạo tập validation (dev) bằng cách tách 10% từ tập train gốc...")
    train_split = dataset['train'].train_test_split(test_size=0.1, seed=42)
    
    print("Đang xử lý tập train (90%)...")
    train_ds = train_split['train'].map(format_sample)
    
    print("Đang xử lý tập validation (10%, dùng làm dev trong lúc train)...")
    dev_ds = train_split['test'].map(format_sample)
    
    print("Đang xử lý tập test (dùng làm eval cuối cùng)...")
    test_ds = dataset['test'].map(format_sample)
    
    train_output = "data/medqa_data/train_data"
    eval_output = "data/medqa_data/eval_data"
    dev_output = "data/medqa_data/dev_data"
    
    os.makedirs("data/medqa_data", exist_ok=True)
    
    train_ds.save_to_disk(train_output)
    test_ds.save_to_disk(eval_output)
    dev_ds.save_to_disk(dev_output)
    
    print(f"✅ Đã tải và lưu tập train tại: {train_output} (Số lượng: {len(train_ds)})")
    print(f"✅ Đã tải và lưu tập eval (test) tại: {eval_output} (Số lượng: {len(test_ds)})")
    print(f"✅ Đã tải và lưu tập dev (validation) tại: {dev_output} (Số lượng: {len(dev_ds)})")

if __name__ == '__main__':
    prepare_medqa()
