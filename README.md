# Optimizing Mathematical Reasoning Performance of Small-Scale Language Models via CoT-SFT-GRPO

**Authors:** Neural Overthinkers

This project implements a robust two-stage training pipeline designed to enhance the mathematical reasoning capabilities of small-scale Language Models (specifically Qwen2.5-Coder-1.5B). By integrating Supervised Fine-Tuning (SFT) with Chain-of-Thought (CoT) data and Group Relative Policy Optimization (GRPO), our experiments verify that prior SFT effectively stabilizes the GRPO training process, mitigating the limitations of individual methods: the shallow pattern imitation of SFT and the training instability of pure RL.

![Optimizing Mathematical Reasoning Poster](./ResultPoster.png)

*Figure — Poster: Optimizing Mathematical Reasoning Performance of Small-Scale LLMs via CoT-SFT-GRPO*

---

## 🌟 Key Highlights

### 1. Stability via SFT Initialization
Our experiments demonstrate that **SFT initialization is essential for stable GRPO training**. It constrains the policy within a safe region of the distribution space, whereas training GRPO directly from the base model often leads to uncontrolled divergence and failure to converge.

### 2. Superior Performance
The **SFT+GRPO** approach consistently outperforms Baseline, SFT-Only, and GRPO-Only methods.

| Dataset | Method | Accuracy (%) |
| :--- | :--- | :--- |
| **GSM8K** | Baseline | 56.61% |
| | Only SFT | 61.17% |
| | Only GRPO | 67.40% |
| | **SFT + GRPO** | **68.47%** |

### 3. Cross-Task Generalization
Unlike pure RL, which can suffer from catastrophic forgetting, our two-stage approach preserves and even enhances performance on out-of-distribution tasks (ARC-Easy, ARC-Challenge).

---

## 📋 Experimental Design

### Ablation Study Matrix

| ID | Method | Training Script | Evaluation Script | Model Architecture | Output Directory | Purpose |
|---|---|---|---|---|---|---|
| 0 | **Baseline** | - | `0-qwen-baseline-inference.py` | Base | `inference_results_baseline/` | Zero-shot Baseline |
| 1-2 | **SFT** | `1-qwen-cot-sft.py` | `2-qwen-cot-inference.py` | Base + SFT LoRA | `qwen_peft_sft_lora/`<br>`inference_results_sft/` | Supervised Learning Effect |
| 3-4 | **GRPO (Base)** | `3-5-qwen-grpo-train-multigpu.py` | `4-qwen-grpo-inference.py` | Base + GRPO LoRA | `qwen_grpo_lora_multigpu/`<br>`inference_results_grpo/` | Pure RL Effect |
| 5-6 | **SFT+GRPO** | `5-qwen-sft-grpo-train.py` | `6-qwen-sft-grpo-eval.py` | Base + SFT + GRPO LoRA | `qwen_sft_grpo_lora/`<br>`inference_results_sft_grpo_full/` | Joint Training Synergy |
| 7 | **GRPO-only (ablation)** | - | `7-qwen-sftbased-grpo-only-eval.py` | Base + GRPO LoRA (no SFT) | `inference_results_grpo_only/` | GRPO Independent Capability |

### Training Strategy

- **Scheme A (Scripts 3-4)**: Train GRPO directly from Base model.
- **Scheme B (Scripts 5-7)**: Layered LoRA
  - Load SFT LoRA → `merge_and_unload()` → Train GRPO LoRA on new base.
  - **Advantage**: SFT and GRPO weights are separated, allowing independent evaluation of their contributions.

---

## 🚀 Quick Start

### Environment Setup
```bash
conda activate torch-env
```

### Training Pipeline

#### Individual Submission
```bash
# 1. SFT Training
sbatch 1.slurm

# 2. Evaluate SFT
sbatch 2.slurm

# 3. GRPO Training (from Base)
sbatch 3-5-multigpu.slurm

# 4. Evaluate GRPO
sbatch 4.slurm

# 5. SFT+GRPO Joint Training
sbatch 5.slurm

# 6. Evaluate Full Model
sbatch 6.slurm

# 7. Evaluate GRPO Independent Capability (Ablation)
sbatch 7.slurm
```

#### Batch Submission (Auto-dependency)

**Method 1: Using `--dependency`**
```bash
JOB5=$(sbatch --parsable 5.slurm)
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 6.slurm)
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5 7.slurm)
```

**Method 2: One-click Script**

```bash
# Execute
./submit_567.sh
```


## 📊 Key Metrics Tracking

### Training Phase
- **Script 5 Output**:
  - `reward_statistics.json` - Detailed reward statistics
  - `training_history.json` - Loss and learning rate curves
  - `training_summary.json` - Training configuration and final metrics

### Evaluation Phase
All evaluation scripts output:
- `results.json` - Detailed results for each sample
- `evaluation_summary.json` - Summary statistics
- `error_cases.json` - Error case analysis
- `visualization_data.json` - Visualization data (for plotting)

---

## 🔧 Strategies for Improvement

### 1. Checkpoint Management & Evaluation Strategy

#### Current Strategy (Practical)

**Save during training**:
```python
# Script 5 Config
save_strategy="steps"
save_steps=100              # Save every 100 steps
save_total_limit=5          # Keep latest 5 checkpoints
```

**Load during evaluation**:
- Scripts 6 & 7 automatically load the **latest checkpoint** (referencing logic from scripts 2 & 4).
- No hardcoded paths; intelligently finds `checkpoint-*` directories.



#### Optional Deep Evaluation Schemes

If the latest checkpoint results are unsatisfactory, manually evaluate others:

**Scheme 1: Select based on training logs**
```python
# Check reward stats during training
reward_logs = json.load("qwen_sft_grpo_lora/reward_statistics.json")
accuracies = [log['accuracy'] for log in reward_logs]

# Find checkpoints corresponding to accuracy peaks
peak_steps = find_peaks(accuracies)
# Manually evaluate these peak checkpoints
```

**Scheme 2: Quick Screen + Full Eval**
```bash
# Step 1: Quick screen all checkpoints with 100 samples (2 mins each)
for cp in checkpoint-*; do
    python eval.py --checkpoint=$cp --test_size=100
done

# Step 2: Run full test on Top 3 (3 × 3 hours)
```

**Scheme 3: No Eval, Just Use**
- With reasonable training params, the latest checkpoint is the best choice.
- Saves 27 hours of evaluation time (90% cost).

#### Why can't GRPO automatically select the best like SFT?

**SFT (Script 1) can**:
```python
load_best_model_at_end=True        # ✅ Auto-save best
dataset_kwargs={"val_size": 0.1}   # ✅ Auto-split validation set
```
- Supervised learning is simple; validation only requires calculating loss.

**GRPO (Scripts 3-5, 5) cannot**:
- RL requires generating multiple candidates (`num_generations=4-8`).
- Needs to run reward function for comparison.
- TRL library's `GRPOConfig` does not support `dataset_kwargs`.
- High validation cost (each validation = running full generation pipeline).

**Conclusion**: Keep multiple checkpoints + default to latest = Balance efficiency and flexibility.

### 2. Evaluation Strategy Enhancement
**Current Issues**:
- Only using greedy decoding (`temperature=0.0`).
- GRPO generates multiple candidates during training (`num_generations=4`), but only 1 during eval.

**Improvement Plan**:
- [ ] **Self-Consistency**: Generate 5 times and take majority vote.
  - Proven effective on CoT tasks by Google paper.
  - Implementation: `model.generate(..., do_sample=True, temperature=0.7, num_return_sequences=5)`
  - Extract results from 5 answers and take the mode.
- [ ] **Sampling Evaluation**: `temperature=0.7` to test generation diversity.
- [ ] **Beam Search**: `num_beams=3` to compare with greedy decoding.

**Note**: Multiple generations during GRPO training are for reward comparison; evaluation can use different strategies.

### 3. Reward Function Optimization
**Current Issues**:
- Binary reward (0/1) is a sparse signal.
- No format reward (cannot distinguish format error vs answer error).

**Improvement Plan**:
- [ ] **Tiered Rewards**:
  ```python
  if format_correct:
      if answer_correct: reward = 1.0
      else: reward = 0.3  # Format correct but answer wrong
  else:
      reward = 0.0  # Format wrong
  ```
- [ ] **Partial Correctness Reward**: Order of magnitude correct gets 0.5.


### 5. Validation Set Evaluation
**Current Missing**:
- No evaluation on validation set during training.
- Cannot monitor overfitting.

**Improvement Plan**:
- [ ] Evaluate accuracy on validation set (50 samples) every N steps.
- [ ] Record validation curve for Early Stopping.
- [ ] Compare training/validation accuracy to monitor generalization.

### 6. Deep Error Analysis
**Current Analysis**:
- Only split into "Format Error" and "Total Error".

**Improvement Plan**:
- [ ] **Problem Type Classification**: Percentage, Fraction, Geometry, Multi-step reasoning, etc.
- [ ] **Error Pattern Recognition**: Calculation error, Understanding error, Unit conversion error.
- [ ] **Visualize Error Distribution**: Group by problem length, complexity.

---
