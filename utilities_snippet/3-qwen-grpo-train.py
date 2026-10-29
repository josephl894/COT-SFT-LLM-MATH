# mamba activate torch-env

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import GRPOTrainer, GRPOConfig
import torch
import time

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
    print(f"GPU count: {torch.cuda.device_count()}")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU name: {gpu_name}")
    print(f"Current device: {torch.cuda.current_device()}")
    
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
    - prompt: question text (the field GRPO requires; not query)
    - ground_truth: correct answer (used by the reward function)
    """
    # Use the new dataset fields directly; no complex parsing
    final_ans = e["answer"].strip()
    
    # GRPO needs a 'prompt' field (not query)
    # also keep ground_truth for the reward function
    return {
        "prompt": e['question'],  # the field name the GRPO trainer expects
        "ground_truth": final_ans  # used by the reward function to check the answer
    }

with Timer("data preprocessing"):
    ds = ds.map(to_chat)

# 3. Load the Qwen-2.5-1.5B model and tokenizer
model_id = "Qwen/Qwen2.5-Coder-1.5B"

with Timer("model load"):
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
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
    # for a small model and for tests, the main linear layers are enough
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],  # includes the MLP layers
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. GRPO training arguments (tuned from TRL practice)
training_args = GRPOConfig(
    output_dir="qwen_grpo_lora",
    num_train_epochs=1,
    
    # ============================================================================
    # Batch settings (uses the A100 80GB memory)
    # ============================================================================
    per_device_train_batch_size=2,  # raised from 1 to 2 (TRL's suggested value)
    gradient_accumulation_steps=4,   # unchanged; effective batch = 2*4 = 8
    
    # ============================================================================
    # Generation settings (core GRPO arguments)
    # ============================================================================
    num_generations=4,  # generate 4 candidate answers per prompt
    generation_batch_size=4,  # ✅ TRL suggests: set the generation batch size explicitly
    max_completion_length=512,  # long enough for math CoT
    max_prompt_length=1024,
    
    # ============================================================================
    # Sampling parameters (more diverse generations)
    # ============================================================================
    temperature=0.9,  # TRL's suggested value; more exploration
    top_p=0.9,        # nucleus sampling
    top_k=50,         # ✅ add top-k sampling
    
    # ============================================================================
    # Optimizer settings (RL wants a smaller learning rate)
    # ============================================================================
    learning_rate=5e-5,  # lowered from 2e-4 to 5e-5 (TRL LoRA example)
    warmup_ratio=0.1,    # add warmup (10% of steps) so early training stays stable
    max_grad_norm=1.0,   # ✅ TRL suggests: gradient clipping so training stays stable
    optim="paged_adamw_8bit",
    
    # ============================================================================
    # GRPO-specific arguments
    # ============================================================================
    beta=0.0,  # KL coefficient (0 = no KL penalty; see the DAPO paper)
    
    # ============================================================================
    # Memory settings (4-bit + one GPU)
    # ============================================================================
    gradient_checkpointing=True,  # ✅ TRL strongly suggests this to save memory
    fp16=True,  # use FP16 (matches quantization)
    
    # ============================================================================
    # Faster data loading (multiprocess preprocessing)
    # ============================================================================
    dataloader_num_workers=4,  # use 4 of the 8 CPUs to load data faster
    dataloader_pin_memory=True,  # faster CPU-to-GPU copies
    
    # ============================================================================
    # Logging and checkpointing
    # ============================================================================
    logging_steps=5,     # log more often (changed from 10 to 5)
    save_strategy="steps",    # save by step, not by epoch
    save_steps=50,            # save every 50 steps
    save_total_limit=3,       # keep only the latest 3 checkpoints
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
    """
    import re
    
    rewards = []
    
    for i, completion in enumerate(completions):
        # default reward is 0 (wrong)
        reward = 0.0
        
        # ground_truth is required in order to score
        if ground_truth is None or i >= len(ground_truth):
            # with no reference answer, right and wrong cannot be judged, so use a neutral reward
            rewards.append(0.5)
            continue
        
        # 1. Extract the final answer from the model output
        # Prefer the text after </think>; otherwise use the whole string
        predicted_text = completion
        if "</think>" in completion:
            # If there is a CoT tag, take the answer after it
            predicted_text = completion.split("</think>")[-1].strip()
        
        # 2. Extract numbers from the answer (GSM8K answers are usually numbers)
        # Regex-extract every number (including negatives and decimals)
        predicted_numbers = re.findall(r'-?\d+\.?\d*', predicted_text)
        gt = str(ground_truth[i]).strip()
        gt_numbers = re.findall(r'-?\d+\.?\d*', gt)
        
        # 3. Binary decision: correct vs wrong
        if predicted_numbers and gt_numbers:
            # Compare the last number (usually the final answer)
            if predicted_numbers[-1] == gt_numbers[-1]:
                reward = 1.0  # ✅ answer correct
            else:
                reward = 0.0  # ❌ answer wrong
        else:
            # If no number can be extracted, try a direct string match
            if gt.lower() in predicted_text.lower():
                reward = 1.0  # ✅ text match succeeded
            else:
                reward = 0.0  # ❌ no match
        
        rewards.append(float(reward))
    
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
        processing_class=tok, # TRL 0.23.0 recommends processing_class
        reward_funcs=reward_func, # Note: the argument is reward_funcs (plural), not reward_fn
    )

print("\n" + "="*80)
print("🚀 Starting GRPO training...")
print(f"📊 Config: {len(ds)} samples × {training_args.num_generations} generations = {len(ds) * training_args.num_generations}  generations")
print(f"💾 Batch size: {training_args.per_device_train_batch_size} × {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps} (effective)")
print("="*80 + "\n")

with Timer("full GRPO training"):
    trainer.train()
