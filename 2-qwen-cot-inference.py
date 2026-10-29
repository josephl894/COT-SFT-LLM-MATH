# mamba activate torch-env

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import json
import re
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
# Small-batch test (100 samples):
# ds = load_dataset("ankner/gsm8k-CoT", split="test[:100]")

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

# 3. Load the base model
print("Loading base model...")
model_id = "Qwen/Qwen2.5-Coder-1.5B"
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)

# 4. Load the LoRA weights
print("Loading LoRA adapter...")

# Find the latest checkpoint automatically
import glob
base_lora_dir = "qwen_peft_sft_lora"

# Find every checkpoint-* directory
checkpoint_dirs = glob.glob(os.path.join(base_lora_dir, "checkpoint-*"))

if checkpoint_dirs:
    # Sort checkpoints by number and take the largest (latest)
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
    lora_path = checkpoint_dirs[-1]  # last = highest number = latest
    print(f"Found {len(checkpoint_dirs)} checkpoints, using latest: {lora_path}")
else:
    # If there is no checkpoint subdirectory, try the root directory
    lora_path = base_lora_dir
    print(f"No checkpoints found, using root directory: {lora_path}")

# Verify the path exists
if not os.path.exists(lora_path):
    print(f"\n{'='*70}")
    print(f"❌ ERROR: LoRA adapter directory not found!")
    print(f"   Looking for: {os.path.abspath(lora_path)}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"\n   Possible solutions:")
    print(f"   1. Run this script from the same directory where training was done")
    print(f"   2. Use absolute path: lora_path = '/home1/yihelu/csci566/qwen_peft_sft_lora'")
    print(f"   3. Run check_model_files.py to diagnose the issue")
    print("="*70)
    import sys
    sys.exit(1)

# Check key files
adapter_config = os.path.join(lora_path, "adapter_config.json")
if not os.path.exists(adapter_config):
    print(f"\n{'='*70}")
    print(f"❌ ERROR: adapter_config.json not found in {lora_path}")
    print(f"   This file is required to load the LoRA adapter.")
    print(f"   Training may not have completed successfully.")
    print("="*70)
    import sys
    sys.exit(1)

print(f"✅ Found LoRA adapter at: {os.path.abspath(lora_path)}")

model = PeftModel.from_pretrained(base_model, lora_path)
model.eval()

# 5. Load the tokenizer
tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tok.pad_token = tok.eos_token

# 6. Inference function
# Note: extract_answer() is already imported from answer_utils; do not redefine it

def generate_answer(question, max_length=2048):
    """
    Generate an answer (deterministic decoding)
    
    ⚠️ Important:The inference prompt must match training!
    - Training used an English prompt with format instructions
    - Inference must use the same format, or quality drops
    
    Generation parameters:
    - temperature=0.0: use greedy decoding (pick the highest-probability token)
    - do_sample=False: turn off random sampling
    - This keeps every run identical, which makes evaluation and comparison easier
    """
    # ============================================================================
    # Format the prompt to match training (scheme 3: <ans> tags)
    # ============================================================================
    prompt = f"""Solve this math problem step by step.

Output format:
1. Wrap reasoning in <think>...</think>
2. Put final answer in <ans>...</ans> (number only, no text)

Example:
<think>
Price: $5
Quantity: 3
Total: $5 × 3 = $15
</think>
<ans>15</ans>

Problem: {question}

Solution:"""
    
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,           # ✅ deterministic generation (greedy decoding)
            do_sample=False,           # ✅ sampling off
            pad_token_id=tok.eos_token_id,
        )
    
    generated_text = tok.decode(outputs[0], skip_special_tokens=True)
    # Drop the input portion
    response = generated_text[len(prompt):].strip()
    return response

# 7. Batch inference
print("Starting inference...")
results = []
correct = 0
total = 0

for i, example in enumerate(tqdm(ds)):
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
        print(f"\nProgress: {i+1}/{len(ds)}, Accuracy so far: {acc:.4f}")

# 8. Compute evaluation metrics
accuracy = correct / total
print(f"\n{'='*50}")
print(f"Final Evaluation Results:")
print(f"Total samples: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"{'='*50}\n")

# 9. Save results
output_dir = "inference_results_sft"  # marked as inference results for the SFT model
os.makedirs(output_dir, exist_ok=True)

# Save detailed inference results
results_file = os.path.join(output_dir, "inference_results.json")
with open(results_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Inference results saved to: {results_file}")

# Save evaluation metrics
metrics = {
    "total_samples": total,
    "correct_predictions": correct,
    "accuracy": accuracy,
    "model_path": lora_path,
    "dataset": "ankner/gsm8k-CoT",
    "split": "test",
}

metrics_file = os.path.join(output_dir, "evaluation_metrics.json")
with open(metrics_file, "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(f"Evaluation metrics saved to: {metrics_file}")

# 10. Save the error-case analysis
wrong_cases = [r for r in results if not r["is_correct"]]
wrong_cases_file = os.path.join(output_dir, "wrong_cases.json")
with open(wrong_cases_file, "w", encoding="utf-8") as f:
    json.dump(wrong_cases, f, ensure_ascii=False, indent=2)
print(f"Wrong cases saved to: {wrong_cases_file}")
print(f"\nTotal wrong cases: {len(wrong_cases)}")

print("\nInference and evaluation completed!")
