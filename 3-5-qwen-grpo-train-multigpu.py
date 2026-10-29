# mamba activate torch-env
# Multi-GPU GRPO training (via accelerate)
# 
# ============================================================================
# Important: how to launch 4-bit quantization + multi-GPU DDP training
# ============================================================================
# 
# ❌ Wrong way (causes a device-mismatch error):
#    python 3-5-qwen-grpo-train-multigpu.py
# 
# ✅ Correct way 1 (preferred, use the config file):
#    accelerate launch --config_file accelerate_config_2gpu.yaml 3-5-qwen-grpo-train-multigpu.py
# 
# ✅ Correct way 2 (pass flags on the command line):
#    accelerate launch --multi_gpu --num_processes=2 3-5-qwen-grpo-train-multigpu.py
# 
# ⚠️ Why LOCAL_RANK is required:
# - accelerate launch starts a separate Python process per GPU
# - Process 0: LOCAL_RANK=0 → GPU 0
# - Process 1: LOCAL_RANK=1 → GPU 1
# - A 4-bit model must be bound to the right device at load time
# - device_map={'': LOCAL_RANK} loads each process's model onto its GPU
# 
# ============================================================================

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import torch
import time
import os
import numpy as np
import json
from datetime import datetime

# Import the shared answer extraction and comparison helpers
from answer_utils import compare_answers, parse_answer_field, extract_number_from_text

# ============================================================================
# Reward-statistics log (for monitoring training)
# ============================================================================
reward_call_count = 0
reward_logs = []

# ============================================================================
# Timing helper
# ============================================================================
class Timer:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.start = time.time()
        print(f"\n⏱️  [{self.name}] starting...")
        return self
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        print(f"✅ [{self.name}] done, elapsed: {elapsed:.2f}s ({elapsed/60:.2f}min)")
        return False

# Check GPU availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    gpu_count = torch.cuda.device_count()
    print(f"GPU count: {gpu_count}")
    for i in range(gpu_count):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Check for an incompatible P100
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        if "P100" in gpu_name:
            print("\n" + "="*70)
            print(f"❌ ERROR: GPU {i} is P100 (CUDA 6.0), not compatible with PyTorch 2.8+")
            print("   Minimum requirement: CUDA capability 7.0 (V100 or newer)")
            print("   Please resubmit the job to get different GPUs.")
            print("="*70)
            import sys
            sys.exit(1)
else:
    print("WARNING: CUDA not available, will use CPU (very slow!)")

# 1. Configure 4-bit quantization
with Timer("quantization config"):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # use NF4 quantization
        bnb_4bit_compute_dtype=torch.float16,  # use float16 (matches fp16 training)
        bnb_4bit_use_double_quant=True,
    )

# 2. Prepare the data - use response as the CoT trace and answer as the final answer
with Timer("dataset load"):
    ds = load_dataset("ankner/gsm8k-CoT", split="train")
    # Small-batch test (10 samples):
    # ds = load_dataset("ankner/gsm8k-CoT", split="train[:10]")

def to_chat(e):
    """
    Convert the dataset into the GRPO training format
    Input fields (ankner/gsm8k-CoT):
    - question: question text (string)
    - response: CoT reasoning steps (string)
    - answer: final answer (string)
    
    Output format:
    - prompt: question text + format instructions (required GRPO field)
    - ground_truth: correct answer (used by the reward function)
    """
    # Use the new dataset fields directly; no complex parsing
    final_ans = e["answer"].strip()
    
    # ============================================================================
    # Format-instruction prompt: scheme 3, <ans> tags
    # ============================================================================
    # Benefits:
    # - Clear split: <think> for reasoning, <ans> for the answer
    # - Extraction is simple: match <ans>...</ans>
    # - Tolerant: the fallback stays in place
    # - Token-efficient (~110 tokens)
    # ============================================================================
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
        "ground_truth": final_ans
    }

with Timer("data preprocessing"):
    ds = ds.map(to_chat)

# 3. Load the Qwen-2.5-1.5B model and tokenizer
model_id = "Qwen/Qwen2.5-Coder-1.5B"

with Timer("model load"):
    # ⚠️ Special requirements for 4-bit + multi-GPU training (TRL docs):
    # Scheme 1: use device_map={'': int(os.environ.get("LOCAL_RANK", 0))}
    # Scheme 2: skip device_map and let accelerate.prepare() place the model
    # This file uses scheme 1 so each process loads the model onto the right GPU
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={'': local_rank},  # ✅ use the LOCAL_RANK environment variable
        torch_dtype=torch.float16,  # match the quantization and training dtype
    )
    tok = AutoTokenizer.from_pretrained(model_id)
    # Make sure Qwen's pad_token is set
    tok.pad_token = tok.eos_token
    model.config.use_cache = False

# 4. Configure LoRA (TRL practice)
lora_config = LoraConfig(
    r=16, # rank r
    lora_alpha=32, # alpha (usually 2x r)
    # ✅ TRL suggests target_modules="all-linear" for better performance
    # for a small model and for tests, naming the modules keeps the parameter count down
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],  # includes the MLP layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. GRPO training arguments (tuned from TRL practice)
training_args = GRPOConfig(
    output_dir="qwen_grpo_lora_multigpu",
    num_train_epochs=1,
    
    # ============================================================================
    # Batch settings (per GPU, under multi-GPU)
    # ============================================================================
    per_device_train_batch_size=2,  # batch size per GPU
    gradient_accumulation_steps=2,   # fewer accumulation steps (the multi-GPU batch is already larger)
    # With 2 GPUs: total batch = 2 GPU × 2 batch × 2 accum = 8
    # With 4 GPUs: total batch = 4 GPU × 2 batch × 2 accum = 16
    
    # ============================================================================
    # Generation settings (core GRPO arguments)
    # ============================================================================
    num_generations=4,  # generate 4 candidate answers per prompt
    generation_batch_size=4,  # ✅ TRL suggests: set the generation batch size explicitly
    max_completion_length=256,  # shorter length to go faster (for testing)
    max_prompt_length=1024,
    
    # ============================================================================
    # Sampling parameters (more diverse generations)
    # ============================================================================
    temperature=0.9,  # TRL suggests 0.7-0.9
    top_p=0.9,
    top_k=50,  # ✅ add top-k sampling
    
    # ============================================================================
    # Optimizer settings (GRPO/RL needs a careful setup)
    # ============================================================================
    learning_rate=5e-5,  # from the TRL GRPO example
    warmup_ratio=0.1,  # warmup for 10% of steps
    max_grad_norm=1.0,  # ✅ TRL suggests: gradient clipping so training stays stable
    optim="paged_adamw_8bit",
    
    # ============================================================================
    # GRPO-specific arguments
    # ============================================================================
    beta=0.01,  # KL coefficient (fixed 2025-11-06: 0.0→0.01)
    # 
    # Why this was changed:
    # - beta=0.0 lets the model drift away from the pretrained distribution with no constraint
    # - GRPO straight from the pretrained model (no SFT) needs some constraint
    # - beta=0.01 is the suggested value when starting from pretraining (a small penalty: explore, but do not collapse)
    # 
    # Tuning notes:
    # - If the output is messy → raise beta (0.05, 0.1)
    # - If the model is too conservative to learn → lower beta (0.001, 0.005)
    # 
    # loss_type="grpo",  # the default; you can leave it unset
    
    # ============================================================================
    # Memory settings (multi-GPU + 4-bit)
    # ============================================================================
    # ⚠️ DDP + gradient_checkpointing + LoRA has a compatibility problem
    # Error: RuntimeError: Expected to mark a variable ready only once
    # 
    # Fix: turn gradient_checkpointing off (required in DDP mode)
    # - DDP uses hooks to track parameter gradients
    # - gradient_checkpointing recomputes and marks parameters more than once
    # - Combined with LoRA parameters, that triggers the conflict
    # 
    # If you run out of memory:
    # - lower per_device_train_batch_size
    # - raise gradient_accumulation_steps
    # - lower num_generations or max_completion_length
    gradient_checkpointing=False,  # ❌ must be off in DDP+LoRA mode
    fp16=True,  # matches the quantization dtype
    
    # ============================================================================
    # Faster data loading
    # ============================================================================
    dataloader_num_workers=4,
    dataloader_pin_memory=True,
    
    # ============================================================================
    # Logging and checkpointing
    # ============================================================================
    logging_steps=2,  # log more often under multi-GPU
    save_strategy="steps",
    save_steps=100,       # save every 100 steps
    save_total_limit=10,  # keep the latest 10 checkpoints (for the training curve)
    
    # ============================================================================
    # Distributed training (accelerate handles this)
    # ============================================================================
    # accelerate detects multi-GPU from the environment
)

# 6. Define the reward function (required by GRPO)
def reward_func(completions, ground_truth=None, **kwargs):
    """
    GRPO reward: score only whether the final answer is correct
    
    Design:
    - each output gets one scalar reward (outcome-based)
    - score only the final result, not the reasoning, format, length, or other intermediate features
    - GRPO normalizes automatically: r̃ᵢ = (rᵢ - mean(r)) / std(r)
    - the normalized reward is assigned to every token of that output
    
    Arguments:
    - completions: list of generated texts (list of strings)
    - ground_truth: list of correct answers (required, for scoring)
    - **kwargs: other arguments (prompts, completions_ids, and so on)
    
    Returns: 
    - List[float]: one scalar score per completion
        * 1.0: answer fully correct
        * 0.0: answer wrong
    
    How the reward works:
    With several samples (num_generations=4), GRPO compares the candidates' rewards:
    - high-reward samples (correct answer) → raise their probability
    - low-reward samples (wrong answer) → lower their probability
    the model learns which reasoning paths lead to the correct answer, without a hand-designed process reward
    
    Note: use the shared answer helpers (answer_utils) so training matches inference
    """
    global reward_call_count, reward_logs
    reward_call_count += 1
    
    rewards = []
    
    for i, completion in enumerate(completions):
        # default reward is 0 (wrong)
        reward = 0.0
        
        # ground_truth is required in order to score
        if ground_truth is None or i >= len(ground_truth):
            # with no reference answer, right and wrong cannot be judged, so use a neutral reward
            rewards.append(0.5)
            continue
        
        # Use the shared answer comparison (numeric)
        # This keeps the training reward identical to the inference accuracy check
        if compare_answers(completion, ground_truth[i]):
            reward = 1.0  # ✅ answer correct
        else:
            reward = 0.0  # ❌ answer wrong
        
        rewards.append(float(reward))
    
    # ============================================================================
    # Record the reward distribution and save the log (main process only)
    # ============================================================================
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
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
            "num_neutral": int((rewards_array == 0.5).sum()),
        }
        reward_logs.append(log_entry)
        
        # Print statistics every 10 calls
        if reward_call_count % 10 == 0:
            print(f"\n{'='*70}")
            print(f"📊 Reward Statistics (Call #{reward_call_count}):")
            print(f"   Mean: {mean_reward:.4f} | Std: {std_reward:.4f} | Acc: {accuracy:.2%}")
            print(f"   Correct: {int(accuracy*len(rewards))}/{len(rewards)}")
            print(f"{'='*70}\n")
        
        # Save the log every 50 calls
        if reward_call_count % 50 == 0:
            output_dir = "qwen_grpo_lora_multigpu"
            os.makedirs(output_dir, exist_ok=True)
            log_file = os.path.join(output_dir, "reward_statistics.json")
            with open(log_file, "w") as f:
                json.dump(reward_logs, f, indent=2)
            print(f"💾 Reward logs saved to: {log_file}")
    
    # Return the raw reward; GRPO normalizes it:
    # r̃ᵢ = (rᵢ - mean(r)) / std(r)
    # then assign r̃ᵢ to every token of output oᵢ as the advantage
    return rewards

# 7. Initialize and start training
with Timer("trainer init"):
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        peft_config=lora_config,
        processing_class=tok,
        reward_funcs=reward_func,
    )

print("\n" + "="*80)
print("🚀 Starting multi-GPU GRPO training...")
gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"🎮 GPU count: {gpu_count}")
print(f"📊 Config: {len(ds)} samples × {training_args.num_generations} generations = {len(ds) * training_args.num_generations}  generations")
total_batch = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * gpu_count
print(f"💾 Total batch size: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} × {gpu_count} = {total_batch}")
print("="*80 + "\n")

with Timer("full GRPO training"):
    trainer.train()

# ============================================================================
# Save the final reward-statistics log
# ============================================================================
if int(os.environ.get("LOCAL_RANK", "0")) == 0 and reward_logs:
    output_dir = "qwen_grpo_lora_multigpu"
    os.makedirs(output_dir, exist_ok=True)
    log_file = os.path.join(output_dir, "reward_statistics.json")
    with open(log_file, "w") as f:
        json.dump(reward_logs, f, indent=2)
    print(f"\n💾 Final reward logs saved to: {log_file}")
    print(f"📊 Total reward function calls: {reward_call_count}")
    print(f"📊 Total log entries: {len(reward_logs)}")

print("\n✅ Training finished. Checkpoints saved in: qwen_grpo_lora_multigpu/")
