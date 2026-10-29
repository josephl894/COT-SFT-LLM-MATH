# mamba activate torch-env
# Baseline eval: use the original Qwen2.5-Coder-1.5B model, with no fine-tuning

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import json
from tqdm import tqdm
import os

# Import the shared answer extraction and comparison helpers
from answer_utils import extract_answer, compare_answers

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU name: {gpu_name}")
    
    # Check for an incompatible P100
    if "P100" in gpu_name:
        print("\n" + "="*70)
        print("❌ ERROR: P100 GPU (CUDA 6.0) is not compatible with PyTorch 2.8+")
        print("   Minimum requirement: CUDA capability 7.0 (V100 or newer)")
        print("   Please resubmit the job to get a different GPU.")
        print("="*70)
        import sys
        sys.exit(1)
else:
    print("WARNING: CUDA not available, will use CPU (very slow!)")

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# 2. Load the dataset (test set: 1320 samples)
print("Loading dataset...")
ds = load_dataset("ankner/gsm8k-CoT", split="test")
# Small-batch test (200 samples):
# ds = load_dataset("ankner/gsm8k-CoT", split="test[:200]")

def to_chat(e):
    """
    Convert the dataset into the inference format
    Input fields (ankner/gsm8k-CoT):
    - question: question text (string)
    - response: CoT reasoning steps (string)
    - answer: final answer (string)
    
    Output format:
    - question: question text
    - answer: <think>reasoning steps</think>\nfinal answer
    - ground_truth: final answer (used for evaluation)
    """
    # Use the new dataset fields directly; no complex parsing
    think_process = e["response"].strip()
    final_ans = e["answer"].strip()
    
    # Build the full answer
    full_answer = f"<think>{think_process}</think>\n{final_ans}"
    ground_truth = final_ans  # reference answer used for evaluation
    
    return {
        "question": e['question'], 
        "answer": full_answer,
        "ground_truth": ground_truth
    }

ds = ds.map(to_chat)
print(f"Loaded {len(ds)} test samples")

# 3. Load the original base model (no LoRA adapter)
print("\n" + "="*70)
print("🔧 Loading BASELINE model (no fine-tuning)...")
print("   Model: Qwen/Qwen2.5-Coder-1.5B")
print("   Quantization: 4-bit (for memory efficiency)")
print("   Fine-tuning: None (this is the baseline)")
print("="*70 + "\n")

model_id = "Qwen/Qwen2.5-Coder-1.5B"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()

# 4. Load the tokenizer
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tok.pad_token = tok.eos_token

# 5. Inference function
def generate_answer(question, max_length=2048):
    """
    Generate an answer (baseline uses Qwen's official chat template)
    
    Design notes:
    - The baseline should show the pretrained model's best performance
    - Use the official chat template so the model works in the format it knows best
    - Do not force a particular output format; let the model answer naturally
    
    Generation parameters:
    - temperature=0.0: use greedy decoding (pick the highest-probability token)
    - do_sample=False: turn off random sampling
    - This keeps every run identical, which makes evaluation and comparison easier
    """
    # Use Qwen's official chat template
    # Per the Qwen2.5-Coder docs, every Qwen model supports apply_chat_template
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant that solves math problems step by step."
        },
        {
            "role": "user",
            "content": f"Solve this problem:\n{question}\n\nPlease think step by step and provide your final numerical answer."
        }
    ]
    
    # Apply the chat template (supported natively by Qwen2.5-Coder)
    text = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,           # ✅ deterministic generation (greedy decoding)
            do_sample=False,           # ✅ sampling off
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )
    
    # Keep only the newly generated text (drop the prompt)
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tok.decode(generated_ids, skip_special_tokens=True).strip()
    
    return response

# 6. Batch inference
print("\n" + "="*70)
print("🚀 Starting BASELINE model inference...")
print(f"📊 Test samples: {len(ds)}")
print(f"🎯 Evaluation metric: Accuracy")
print(f"🔧 Generation mode: Deterministic (temperature=0.0)")
print("="*70 + "\n")

results = []
correct = 0
total = 0

for i, example in enumerate(tqdm(ds, desc="Evaluating")):
    question = example["question"]
    ground_truth = example["ground_truth"]
    
    # Generate an answer
    generated_response = generate_answer(question)
    predicted_answer = extract_answer(generated_response)
    gt_answer = extract_answer(ground_truth)
    
    # Decide correctness (numeric comparison)
    is_correct = compare_answers(predicted_answer, gt_answer)
    if is_correct:
        correct += 1
    total += 1
    
    # Save results
    result = {
        "index": i,
        "question": question,
        "ground_truth": ground_truth,
        "ground_truth_extracted": gt_answer,
        "generated_response": generated_response,
        "predicted_answer": predicted_answer,
        "is_correct": is_correct
    }
    results.append(result)
    
    # Print progress every 50 samples
    if (i + 1) % 50 == 0:
        acc = correct / total
        print(f"\nProgress: {i+1}/{len(ds)}, Accuracy so far: {acc:.4f} ({acc*100:.2f}%)")

# 7. Compute evaluation metrics
accuracy = correct / total
print(f"\n{'='*70}")
print(f"📈 Final Evaluation Results (BASELINE Model):")
print(f"{'='*70}")
print(f"Total samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Wrong predictions: {total - correct}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*70}\n")

# 8. Save results
output_dir = "inference_results_baseline"  # inference results for the baseline model
os.makedirs(output_dir, exist_ok=True)

# Save detailed inference results
results_file = os.path.join(output_dir, "inference_results.json")
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"✅ Inference results saved to: {results_file}")

# Save evaluation metrics
metrics = {
    "total_samples": total,
    "correct_predictions": correct,
    "wrong_predictions": total - correct,
    "accuracy": accuracy,
    "model_type": "BASELINE",
    "model_path": model_id,
    "base_model": model_id,
    "fine_tuning": "None",
    "dataset": "ankner/gsm8k-CoT",
    "split": "test",
    "generation_config": {
        "max_new_tokens": 512,
        "temperature": 0.0,
        "do_sample": False,
        "method": "greedy_decoding"
    }
}

metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"✅ Evaluation metrics saved to: {metrics_file}")

# 9. Save the error-case analysis
wrong_cases = [r for r in results if not r["is_correct"]]
wrong_cases_file = os.path.join(output_dir, "wrong_cases.json")
with open(wrong_cases_file, "w", encoding="utf-8") as f:
    json.dump(wrong_cases, f, ensure_ascii=False, indent=2)
print(f"✅ Wrong cases saved to: {wrong_cases_file}")
print(f"   Total wrong cases: {len(wrong_cases)}")

# 10. Print a few example results
if len(results) > 0:
    print(f"\n{'='*70}")
    print("📝 Sample Results:")
    print(f"{'='*70}")
    
    # Print the first correct case
    correct_cases = [r for r in results if r["is_correct"]]
    if correct_cases:
        sample = correct_cases[0]
        print(f"\n✅ Correct Example (Index {sample['index']}):")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Predicted: {sample['predicted_answer']}")
        print(f"Ground Truth: {sample['ground_truth_extracted']}")
    
    # Print the first incorrect case
    if wrong_cases:
        sample = wrong_cases[0]
        print(f"\n❌ Wrong Example (Index {sample['index']}):")
        print(f"Question: {sample['question'][:100]}...")
        print(f"Predicted: {sample['predicted_answer']}")
        print(f"Ground Truth: {sample['ground_truth_extracted']}")
        print(f"Full Response: {sample['generated_response'][:200]}...")

print(f"\n{'='*70}")
print("🎉 BASELINE Model Inference and Evaluation Completed!")
print(f"{'='*70}")
print("\n💡 This baseline result can be compared with:")
print("   - SFT model: inference_results_sft/")
print("   - GRPO model: inference_results_grpo/")
print(f"{'='*70}\n")
