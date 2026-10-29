# mamba activate torch-env
import torch
import os
import json
import time
import re
import glob
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from answer_utils import extract_answer, compare_answers
import numpy as np
from datetime import datetime

# ============================================================================
# Config
# ============================================================================
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B"

# Load the training output for TRAINING_JOB_ID (parallel evaluation)
training_job_id = os.environ.get("TRAINING_JOB_ID", "default")
GRPO_BASE_DIR = f"./qwen_sft_grpo_lora_{training_job_id}"  # GRPO LoRA base directory
OUTPUT_DIR = f"./inference_results_grpo_only_{training_job_id}"  # evaluation output directory

print(f"\n📁 Evaluating Training Job ID: {training_job_id}")
print(f"📁 Loading GRPO from: {GRPO_BASE_DIR}")
print(f"📁 Saving results to: {OUTPUT_DIR}")

# 1. Data loading
# 🚀 Production mode (full test set: 1319 rows)
ds = load_dataset("ankner/gsm8k-CoT", split="test")
# 🧪 Test mode (10 samples) — for debugging
# ds = load_dataset("ankner/gsm8k-CoT", split="test[:10]")

print(f"Loaded {len(ds)} test samples")

# 2. Load the model (GRPO LoRA only, no SFT)
print("=" * 70)
print("🔬 Evaluation Mode: GRPO-only (Ablation Study)")
print("=" * 70)
print("Loading Base Model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

# ============================================================================
# Important: load the GRPO LoRA from script 5 directly onto the original base
# ============================================================================
# Goal: test the GRPO LoRA on its own (without the SFT base)
# 
# Note: script 5 trained its GRPO LoRA on a merged (Base+SFT) model,
#       here we deliberately skip SFT, to see what GRPO does alone
# ============================================================================
print(f"Searching for GRPO checkpoints in: {GRPO_BASE_DIR}")
grpo_checkpoints = glob.glob(os.path.join(GRPO_BASE_DIR, "checkpoint-*"))

if grpo_checkpoints:
    # Sort checkpoints by number and take the latest
    grpo_checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    GRPO_ADAPTER_PATH = grpo_checkpoints[-1]
    print(f"Found {len(grpo_checkpoints)} GRPO checkpoints, using latest: {GRPO_ADAPTER_PATH}")
elif os.path.exists(os.path.join(GRPO_BASE_DIR, "adapter_config.json")):
    GRPO_ADAPTER_PATH = GRPO_BASE_DIR
    print(f"Using GRPO adapter from root directory: {GRPO_ADAPTER_PATH}")
else:
    print(f"❌ No GRPO checkpoints found in {GRPO_BASE_DIR}!")
    exit(1)

print(f"Loading GRPO Adapter (without SFT) from: {GRPO_ADAPTER_PATH}")
print("⚠️  Note: This GRPO was trained on (Base+SFT), but we load it on Base only")
print("    to test its independent capability.")
model = PeftModel.from_pretrained(model, GRPO_ADAPTER_PATH)
print("✅ GRPO Adapter loaded.")
print("📊 Final Model Architecture: Base + GRPO LoRA (without SFT)")

model.eval()
tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tok.pad_token = tok.eos_token

# 3. Inference function (records detailed metrics)
def generate_answer(question):
    """
    Generate an answer and record performance metrics
    
    Returns:
    - response: generated text
    - metrics: dict of metrics such as inference time and token count
    """
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
    
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    prompt_length = inputs['input_ids'].shape[1]
    
    # Record inference time
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=512, 
            temperature=0.0, # greedy decoding for evaluation
            do_sample=False
        )
    
    inference_time = time.time() - start_time
    
    # Decode only the generated part
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tok.decode(generated_ids, skip_special_tokens=True)
    
    # Compute generation metrics
    generation_length = len(generated_ids)
    tokens_per_second = generation_length / inference_time if inference_time > 0 else 0
    
    # Check whether the format was followed
    has_think_tag = '<think>' in response and '</think>' in response
    has_ans_tag = '<ans>' in response and '</ans>' in response
    
    metrics = {
        "inference_time": inference_time,
        "prompt_length": prompt_length,
        "generation_length": generation_length,
        "total_length": prompt_length + generation_length,
        "tokens_per_second": tokens_per_second,
        "has_think_tag": has_think_tag,
        "has_ans_tag": has_ans_tag,
        "format_correct": has_think_tag and has_ans_tag,
    }
    
    return response, metrics

# 4. Batch evaluation (records detailed metrics)
results = []
correct = 0
all_metrics = []
print("🚀 Starting Inference (GRPO-only mode)...")

eval_start_time = time.time()

for i, example in enumerate(tqdm(ds)):
    q = example["question"]
    gt = example["answer"] # answer field from the original dataset
    
    # Generate an answer and collect metrics
    pred_raw, gen_metrics = generate_answer(q)
    pred_ans = extract_answer(pred_raw)
    gt_ans = extract_answer(gt)
    
    is_correct = compare_answers(pred_ans, gt_ans)
    if is_correct: correct += 1
    
    # Merge the result and the metrics
    result_entry = {
        "sample_id": i,
        "question": q,
        "ground_truth": gt_ans,
        "prediction": pred_ans,
        "full_response": pred_raw,
        "correct": is_correct,
        **gen_metrics  # includes every generation metric
    }
    results.append(result_entry)
    all_metrics.append(gen_metrics)
    
    # Print a progress summary every 10 samples
    if (i + 1) % 10 == 0:
        current_acc = correct / (i + 1)
        avg_time = np.mean([m["inference_time"] for m in all_metrics[-10:]])
        print(f"\n  Progress: {i+1}/{len(ds)} | Acc: {current_acc:.2%} | Avg Time: {avg_time:.2f}s")

total_eval_time = time.time() - eval_start_time

# ============================================================================
# Statistics
# ============================================================================
acc = correct / len(ds)

# Performance metrics
inference_times = [m["inference_time"] for m in all_metrics]
generation_lengths = [m["generation_length"] for m in all_metrics]
tokens_per_second = [m["tokens_per_second"] for m in all_metrics]
format_correct = [m["format_correct"] for m in all_metrics]

# Accuracy broken down by generation length
length_bins = [0, 50, 100, 200, 512]
acc_by_length = {}
for i in range(len(length_bins) - 1):
    bin_name = f"{length_bins[i]}-{length_bins[i+1]}"
    samples_in_bin = [r for r in results if length_bins[i] <= r["generation_length"] < length_bins[i+1]]
    if samples_in_bin:
        bin_acc = sum(r["correct"] for r in samples_in_bin) / len(samples_in_bin)
        acc_by_length[bin_name] = {
            "accuracy": bin_acc,
            "count": len(samples_in_bin)
        }

# Error-case analysis
error_cases = [r for r in results if not r["correct"]]
error_by_format = {
    "format_correct": sum(1 for e in error_cases if e["format_correct"]),
    "format_incorrect": sum(1 for e in error_cases if not e["format_correct"]),
}

# Write the evaluation report
evaluation_summary = {
    "training_job_id": training_job_id,
    "timestamp": datetime.now().isoformat(),
    "eval_mode": "grpo_only (ablation study)",
    "model_architecture": "Base + GRPO LoRA (without SFT)",
    "grpo_adapter_path": GRPO_ADAPTER_PATH,
    "note": "GRPO was trained on (Base+SFT) but evaluated on Base only to test independent capability",
    "dataset": "ankner/gsm8k-CoT test split",
    "num_samples": len(ds),
    "total_eval_time": total_eval_time,
    "avg_time_per_sample": total_eval_time / len(ds),
    
    # Accuracy metrics
    "accuracy": acc,
    "num_correct": correct,
    "num_wrong": len(ds) - correct,
    
    # Performance metrics
    "inference_time": {
        "mean": float(np.mean(inference_times)),
        "std": float(np.std(inference_times)),
        "min": float(np.min(inference_times)),
        "max": float(np.max(inference_times)),
        "median": float(np.median(inference_times)),
    },
    
    "generation_length": {
        "mean": float(np.mean(generation_lengths)),
        "std": float(np.std(generation_lengths)),
        "min": int(np.min(generation_lengths)),
        "max": int(np.max(generation_lengths)),
        "median": float(np.median(generation_lengths)),
    },
    
    "tokens_per_second": {
        "mean": float(np.mean(tokens_per_second)),
        "median": float(np.median(tokens_per_second)),
    },
    
    # Format compliance
    "format_compliance": {
        "format_correct_rate": sum(format_correct) / len(format_correct),
        "format_correct_count": sum(format_correct),
    },
    
    # Accuracy breakdown
    "accuracy_by_length": acc_by_length,
    
    # Error analysis
    "error_analysis": {
        "total_errors": len(error_cases),
        "errors_by_format": error_by_format,
    }
}

# ============================================================================
# Print the evaluation results
# ============================================================================
print(f"\n{'='*70}")
print(f"📊 Evaluation Results Summary (GRPO-only)")
print(f"{'='*70}")
print(f"⚠️  Ablation Mode: Testing GRPO LoRA without SFT foundation")
print(f"Overall Accuracy: {acc:.2%} ({correct}/{len(ds)})")
print(f"Total Eval Time: {total_eval_time:.2f}s ({total_eval_time/60:.2f} min)")
print(f"Avg Time/Sample: {evaluation_summary['avg_time_per_sample']:.2f}s")
print(f"\n📈 Generation Metrics:")
print(f"  Avg Length: {evaluation_summary['generation_length']['mean']:.1f} tokens")
print(f"  Avg Speed: {evaluation_summary['tokens_per_second']['mean']:.1f} tok/s")
print(f"  Format Correct: {evaluation_summary['format_compliance']['format_correct_rate']:.2%}")
print(f"\n🔍 Error Analysis:")
print(f"  Total Errors: {len(error_cases)}")
if error_cases:
    print(f"  Errors with correct format: {error_by_format['format_correct']}")
    print(f"  Errors with incorrect format: {error_by_format['format_incorrect']}")
print(f"\n💡 Interpretation:")
print(f"  Low accuracy indicates GRPO LoRA depends on SFT foundation.")
print(f"  Compare with script 6 (full SFT+GRPO) to see the difference.")
print(f"{'='*70}\n")

# ============================================================================
# Save all results
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Detailed results (every sample)
output_file = os.path.join(OUTPUT_DIR, "results.json")
with open(output_file, "w") as f:
    json.dump(results, f, indent=2)
print(f"✅ Detailed results saved to: {output_file}")

# 2. Evaluation summary (for a quick look)
summary_file = os.path.join(OUTPUT_DIR, "evaluation_summary.json")
with open(summary_file, "w") as f:
    json.dump(evaluation_summary, f, indent=2)
print(f"✅ Evaluation summary saved to: {summary_file}")

# 3. Error cases (for analysis)
if error_cases:
    error_file = os.path.join(OUTPUT_DIR, "error_cases.json")
    with open(error_file, "w") as f:
        json.dump(error_cases, f, indent=2)
    print(f"✅ Error cases saved to: {error_file}")

# 4. Plot data
viz_data = {
    "sample_ids": list(range(len(results))),
    "correctness": [int(r["correct"]) for r in results],
    "inference_times": inference_times,
    "generation_lengths": generation_lengths,
    "tokens_per_second": tokens_per_second,
    "format_correct": [int(f) for f in format_correct],
}
viz_file = os.path.join(OUTPUT_DIR, "visualization_data.json")
with open(viz_file, "w") as f:
    json.dump(viz_data, f, indent=2)
print(f"✅ Visualization data saved to: {viz_file}")

print(f"\n📁 All files saved to: {OUTPUT_DIR}/")
