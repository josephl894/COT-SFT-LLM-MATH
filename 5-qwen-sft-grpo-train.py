# mamba activate torch-env
# Multi-GPU GRPO training (via accelerate)
# 
# ============================================================================
# Important: how to launch 4-bit quantization + multi-GPU DDP training
# ============================================================================
# ✅ Correct way (use the config file):
#    accelerate launch --config_file accelerate_config_2gpu.yaml 5-qwen-sft-grpo-train.py
# ============================================================================

import os
import torch
import glob
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig
from trl import GRPOTrainer, GRPOConfig
from answer_utils import compare_answers
import numpy as np
import json
from datetime import datetime

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("WARNING: CUDA not available, will use CPU (very slow!)")

# ============================================================================
# Config: paths and arguments
# ============================================================================
# Input: adapter path from the SFT stage (produced by 1-qwen-cot-sft.py)
SFT_BASE_DIR = "./qwen_peft_sft_lora"  # SFT LoRA base directory

# Output: a unique directory from the Job ID (parallel training)
job_id = os.environ.get("SLURM_JOB_ID", "default")
OUTPUT_DIR = f"./qwen_sft_grpo_lora_{job_id}"
print(f"\n📁 Training Job ID: {job_id}")
print(f"📁 Output Directory: {OUTPUT_DIR}")

# ============================================================================
# Training metrics (for plots)
# ============================================================================
reward_call_count = 0
reward_logs = []
training_metrics = []

# 1. Data preparation
# ----------------------------------------------------------------------------
# 🚀 Production mode (full data: 7470 rows)
ds = load_dataset("ankner/gsm8k-CoT", split="train")

# 🧪 Test mode (10 samples) — for debugging
# ds = load_dataset("ankner/gsm8k-CoT", split="train[:10]")
# ----------------------------------------------------------------------------

def to_chat(e):
    # Same format as SFT, but GRPO needs prompt and ground_truth
    formatted_prompt = f"""Solve this math problem step by step.

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

Problem: {e['question']}

Solution:"""
    return {
        "prompt": formatted_prompt,
        "ground_truth": e["answer"].strip()
    }

ds = ds.map(to_chat)

# 2. Load the model (Base + SFT → merge → ready for GRPO)
model_id = "Qwen/Qwen2.5-Coder-1.5B"

# quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ============================================================================
# Multi-GPU device setup: use the LOCAL_RANK environment variable
# ============================================================================
# accelerate launch starts a separate process per GPU:
# - Process 0: LOCAL_RANK=0 → GPU 0
# - Process 1: LOCAL_RANK=1 → GPU 1
# A 4-bit model must be bound to the right device at load time
local_rank = int(os.environ.get("LOCAL_RANK", 0))
print(f"\n📍 Process LOCAL_RANK: {local_rank}")

print(f"Loading Base Model: {model_id}")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={'': local_rank},  # Multi-GPU: each process loads onto its own GPU
    torch_dtype=torch.float16,
)

# ============================================================================
# Scheme B: layered LoRA training
# ============================================================================
# Step 1: find and load the latest SFT adapter checkpoint
print(f"Searching for SFT checkpoints in: {SFT_BASE_DIR}")
sft_checkpoints = glob.glob(os.path.join(SFT_BASE_DIR, "checkpoint-*"))

if sft_checkpoints:
    # Sort checkpoints by number and take the latest
    sft_checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    SFT_ADAPTER_PATH = sft_checkpoints[-1]
    print(f"Found {len(sft_checkpoints)} SFT checkpoints, using latest: {SFT_ADAPTER_PATH}")
elif os.path.exists(os.path.join(SFT_BASE_DIR, "adapter_config.json")):
    # If the root directory has a config (the best model), use it directly
    SFT_ADAPTER_PATH = SFT_BASE_DIR
    print(f"Using SFT adapter from root directory: {SFT_ADAPTER_PATH}")
else:
    raise FileNotFoundError(f"No SFT checkpoints found in {SFT_BASE_DIR}. Please run script 1 first.")

print(f"Loading SFT Adapter from: {SFT_ADAPTER_PATH}")
model = PeftModel.from_pretrained(model, SFT_ADAPTER_PATH)
print("✅ SFT Adapter loaded.")

# Step 2: merge the SFT weights into the base model (required)
print("Merging SFT adapter into base model...")
model = model.merge_and_unload()
print("✅ SFT weights merged. Now the base model includes SFT knowledge.")
# The model is now Base + SFT (baked in); the original SFT LoRA parameters are no longer needed

tok = AutoTokenizer.from_pretrained(model_id)
tok.pad_token = tok.eos_token
model.config.use_cache = False

# 3. Configure a new LoRA (used only for GRPO)
# This LoRA learns the GRPO update on top of (Base+SFT)
grpo_lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 4. Training arguments
training_args = GRPOConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,
    
    # ========================================================================
    # 🚀 Production settings (full training: 7470 rows)
    # ⚠️  Note: under GRPO + DDP, gradient_accumulation has no effect; use a larger batch_size
    # ========================================================================
    per_device_train_batch_size=4,  # 4 rows per GPU (2 GPUs = 8 rows total)
    gradient_accumulation_steps=1,  # set to 1 under DDP (no gradient accumulation)
    num_generations=8,              # more samples, for a better baseline
    max_completion_length=512,      # allow a longer reasoning chain
    
    # ========================================================================
    # 🧪 Test settings (quick check) — for debugging
    # ========================================================================
    # per_device_train_batch_size=2,
    # gradient_accumulation_steps=2,
    # num_generations=4,
    # max_completion_length=128,
    
    # Shared arguments
    learning_rate=5e-6, # ⚠️ scheme B can use a slightly higher learning rate, because the new LoRA trains on top of baked-in SFT
    beta=0.01,          # KL penalty coefficient
    logging_steps=10,   # log less often under multi-GPU
    save_strategy="steps",
    save_steps=100,      # 🚀 production: save every 100 steps
    save_total_limit=10, # keep the latest 10 checkpoints
    fp16=True,
    gradient_checkpointing=False, # ⚠️ must be off in DDP+LoRA mode
    dataloader_num_workers=4,  # faster multi-GPU data loading
    dataloader_pin_memory=True,
    report_to="none",
)

# 5. Reward function (with detailed stats and a log)
def reward_func(completions, ground_truth=None, **kwargs):
    """
    GRPO reward: score the final answer and record detailed statistics
    
    Logged metrics:
    - mean_reward: mean reward
    - std_reward: reward standard deviation
    - accuracy: accuracy
    - num_correct/wrong: correct/wrong counts
    """
    global reward_call_count, reward_logs
    reward_call_count += 1
    
    rewards = []
    for i, completion in enumerate(completions):
        if ground_truth is None:
            rewards.append(0.0)
            continue
        # Compare with the shared helper
        if compare_answers(completion, ground_truth[i]):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    
    # ============================================================================
    # Record the reward distribution and save the log
    # ============================================================================
    rewards_array = np.array(rewards)
    mean_reward = rewards_array.mean()
    std_reward = rewards_array.std()
    accuracy = (rewards_array == 1.0).mean()
    
    # Append to the JSON log
    log_entry = {
        "call_count": reward_call_count,
        "timestamp": datetime.now().isoformat(),
        "batch_size": len(rewards),
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "accuracy": float(accuracy),
        "num_correct": int((rewards_array == 1.0).sum()),
        "num_wrong": int((rewards_array == 0.0).sum()),
    }
    reward_logs.append(log_entry)
    
    # Print statistics every 5 calls (more often in test mode)
    if reward_call_count % 5 == 0:
        print(f"\n{'='*70}")
        print(f"📊 Reward Statistics (Call #{reward_call_count}):")
        print(f"   Mean: {mean_reward:.4f} | Std: {std_reward:.4f} | Acc: {accuracy:.2%}")
        print(f"   Correct: {int(accuracy*len(rewards))}/{len(rewards)}")
        print(f"{'='*70}\n")
    
    return rewards

# 6. Start training
# ⚠️ Important: pass peft_config so GRPO trains a new LoRA on (Base+SFT)
trainer = GRPOTrainer(
    model=model,  # this base model already includes SFT
    args=training_args,
    train_dataset=ds,
    peft_config=grpo_lora_config,  # the new LoRA layer (the point of scheme B)
    processing_class=tok,
    reward_funcs=reward_func,
)

print("🚀 Starting SFT+GRPO Training (Two-Layer LoRA Approach)...")
print(f"Architecture: Base + SFT (merged) + GRPO LoRA (new)")
print(f"Input SFT Adapter: {SFT_ADAPTER_PATH} (will be merged)")
print(f"Output GRPO Adapter: {OUTPUT_DIR} (incremental LoRA)")

# Train and keep the training history
train_result = trainer.train()

# Save the final model
trainer.save_model(OUTPUT_DIR)

# ============================================================================
# Save training metrics for plots
# ============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Save the reward-statistics log
if reward_logs:
    reward_log_file = os.path.join(OUTPUT_DIR, "reward_statistics.json")
    with open(reward_log_file, "w") as f:
        json.dump(reward_logs, f, indent=2)
    print(f"\n💾 Reward logs saved to: {reward_log_file}")
    print(f"📊 Total reward function calls: {reward_call_count}")

# 2. Save the training history (loss, learning_rate, and so on)
if hasattr(trainer.state, 'log_history') and trainer.state.log_history:
    training_log_file = os.path.join(OUTPUT_DIR, "training_history.json")
    with open(training_log_file, "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)
    print(f"💾 Training history saved to: {training_log_file}")

# 3. Write the training summary
training_summary = {
    "job_id": job_id,
    "start_time": datetime.now().isoformat(),
    "model_id": model_id,
    "sft_adapter_path": SFT_ADAPTER_PATH,
    "output_dir": OUTPUT_DIR,
    "num_samples": len(ds),
    "num_epochs": training_args.num_train_epochs,
    "batch_size": training_args.per_device_train_batch_size,
    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    "learning_rate": training_args.learning_rate,
    "beta": training_args.beta,
    "num_generations": training_args.num_generations,
    "max_completion_length": training_args.max_completion_length,
    "total_reward_calls": reward_call_count,
}

# Add the final reward statistics
if reward_logs:
    final_rewards = [log["accuracy"] for log in reward_logs]
    training_summary["final_avg_accuracy"] = float(np.mean(final_rewards))
    training_summary["final_max_accuracy"] = float(np.max(final_rewards))
    training_summary["final_min_accuracy"] = float(np.min(final_rewards))

summary_file = os.path.join(OUTPUT_DIR, "training_summary.json")
with open(summary_file, "w") as f:
    json.dump(training_summary, f, indent=2)
print(f"💾 Training summary saved to: {summary_file}")

print(f"\n✅ Training finished. Model saved to {OUTPUT_DIR}")
print(f"\n📊 Quick Summary:")
print(f"   Total Reward Calls: {reward_call_count}")
if reward_logs:
    print(f"   Final Accuracy: {training_summary.get('final_avg_accuracy', 0):.2%}")
print(f"   Logs available for visualization in: {OUTPUT_DIR}")
