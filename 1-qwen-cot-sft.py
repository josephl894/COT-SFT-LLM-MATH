# mamba activate torch-env

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import torch

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

# 1. Configure 4-bit quantization (replaces Unsloth's load_in_4bit)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",  # use NF4 quantization
    bnb_4bit_compute_dtype=torch.bfloat16, # compute in bfloat16; V100 supports it
    bnb_4bit_use_double_quant=True,
)

# 2. Prepare the data - use response as the CoT trace and answer as the final answer
# Full training set: 7470 samples
ds = load_dataset("ankner/gsm8k-CoT", split="train")
# Small-batch test (100 samples):
# ds = load_dataset("ankner/gsm8k-CoT", split="train[:100]")

def to_chat(e):
    """
    Convert the dataset into the training format
    Input fields (ankner/gsm8k-CoT):
    - question: question text (string)
    - response: CoT reasoning steps (string)
    - answer: final answer (string)
    
    Output format:
    - question: question text
    - answer: <think>reasoning steps</think>\nfinal answer
    """
    # Use the new dataset fields directly; no complex parsing
    think_process = e["response"].strip()
    final_ans = e["answer"].strip()
    
    # Build the full answer: <think>reasoning</think>\nfinal answer
    full_answer = f"<think>{think_process}</think>\n{final_ans}"
    return {"question": e['question'], "answer": full_answer}

ds = ds.map(to_chat)

# 3. Load the Qwen-2.5-1.5B model and tokenizer
# Load the model with Transformers and quantize to 4-bit via BitsAndBytesConfig
model_id = "Qwen/Qwen2.5-Coder-1.5B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained(model_id)
# Make sure Qwen's pad_token is set
tok.pad_token = tok.eos_token
model.config.use_cache = False

# 4. Configure LoRA (same purpose as the original code)
lora_config = LoraConfig(
    r=16, # rank r
    lora_alpha=32, # alpha
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # target modules
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Training arguments (SFTConfig, recommended by TRL 0.23.0)
training_args = SFTConfig(
    output_dir="qwen_peft_sft_lora",
    num_train_epochs=1,
    per_device_train_batch_size=1, # micro_batch_size
    gradient_accumulation_steps=4, # gradient_accumulation
    learning_rate=2e-4,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True, # V100 does not support bf16; use fp16
    optim="paged_adamw_8bit", # optimizer
    dataset_text_field="text", # name of the text field
    
    # ===== Validation-set settings =====
    # Hold out 10% of the training set as validation
    dataset_kwargs={"val_size": 0.1},
    # Evaluate the validation set every 100 steps
    evaluation_strategy="steps",
    eval_steps=100,
    # Keep the best model (by validation loss)
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Rebuild the data using the standard text field
def prepare_text(e):
    # ============================================================================
    # Scheme 3: <ans> tags — a numbered list plus a clear answer tag
    # ============================================================================
    formatted_question = f"""Solve this math problem step by step.

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
    
    # ⚠️ Important:Add a space after "Solution:" so the format matches inference exactly
    text = f"{formatted_question} {e['answer']}"
    return {"text": text}

ds = ds.map(prepare_text)

# 6. Initialize and start training
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    peft_config=lora_config,
    processing_class=tok, # TRL 0.23.0 prefer processing_class rather than tokenizer
)

trainer.train()