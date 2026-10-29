# 📊 Experiment 3-5: GRPO-only Training Dynamics Analysis and Figure Guide

> **Package Contents:**
> - `training_metrics.csv` - full training-metric data (1863 steps)
> - `training_metrics_visualization.png` - training-dynamics figure
> - `visualize_training_metrics.py` - figure generation script
> - `extract_training_metrics.py` - data extraction script

> **Experiment Characteristics:**
> - **No SFT pretraining** — GRPO training starts directly from the base model
> - **Compared with Experiment 5** — to see the effect of SFT pretraining

---

## 📋 Table of Contents

1. [Experiment Configuration](#experiment-configuration)
2. [Key Metrics Analysis](#key-metrics-analysis)
3. [Comparison with Experiment 5](#comparison-with-experiment-5)
4. [Figure Caption Guide](#figure-caption-guide)
5. [Academic Writing Guidelines](#academic-writing-guidelines)
6. [FAQ](#faq)

---

## 🔧 Experiment Configuration

### Model & Hardware

| Configuration | Value | Detail |
|--------|--------------|-------|
| Base Model | Qwen/Qwen2.5-Coder-1.5B | (1.5B parameters) |
| Quantization | 4-bit NF4 | (BitsAndBytes) |
| GPU Setup | 2× GPU | Multi-GPU DDP |
| Training Dataset | ankner/gsm8k-CoT | 7465 samples |
| Framework | TRL GRPOTrainer | v0.23.0 |
| **Training Strategy** | **GRPO-only** | **No SFT pretraining** |

### Training Hyperparameters

```python
# Key configuration
learning_rate = 5e-5              # Initial LR (larger than Experiment 5)
lr_scheduler = "linear"            # Linear decay
num_train_steps = 3733             # Total steps
per_device_batch_size = 2          # Per-device batch
gradient_accumulation_steps = 2    # Gradient accumulation
num_generations = 4                # Generations/batch
total_batch_size = 8               # Total: 2×2×2
max_completion_length = 256        # Max generation length
beta = 0.01                        # KL penalty coefficient
```

### LoRA Configuration

```python
# Single-layer LoRA strategy
# Train GRPO LoRA directly on the base model

lora_rank = 16                     # Rank
lora_alpha = 32                    # Alpha
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
```

---

## 📊 Key Metrics Analysis

### 1️⃣ Model Performance

**Figure Location:** Subplot (a) - Top main panel

#### 📈 Data Summary

| Metric | Value | Detail |
|------|--------|-------|
| Initial Accuracy | **31.2%** | Training start |
| Final Accuracy | **50.0%** | After 1863 steps |
| Peak Accuracy | **100.0%** | Maximum observed |
| Mean Accuracy | **59.8%** | Overall average |
| Improvement | **+18.8%** (absolute) / **60.0%** (relative) | Absolute / Relative |

#### 🔍 Detailed Analysis

1. **Low Starting Point (Step 0-500):**
   - Accuracy starts at 31.2%, significantly lower than Experiment 5's 49.4%
   - **Reason:** No SFT pretraining, starting from base model
   - High volatility reflects model exploring problem-solving strategies

2. **Rapid Learning Phase (Step 500-1200):**
   - Accuracy rapidly increases to ~55-60%
   - Steeper learning curve than Experiment 5
   - Demonstrates GRPO can learn mathematical reasoning from scratch

3. **Plateau Phase (Step 1200-1863):**
   - Accuracy fluctuates around 50%
   - Does not reach Experiment 5's 71.2% final accuracy
   - Can peak at 100% but unstable

#### ⚠️ Important Note

**This is "In-Training Evaluation on Training Set", NOT independent test set!**

---

### 2️⃣ Training Loss

**Figure Location:** Subplot (b) - Bottom left

#### 📉 Analysis

- Loss ranges from -0.06 to 0.20
- More volatile than Experiment 5
- Final positive loss indicates ongoing training
- Characteristic of GRPO from scratch: unstable loss

---

### 3️⃣ KL Divergence from Base Model

**Figure Location:** Subplot (c) - Bottom center

#### 🎯 Key Findings

| Steps | Stage | KL Value | Level | Status |
|------|-------|------|----------|--------|
| Step 0-500 | Early | 0.00-0.01 | Very Low | ✅ Healthy |
| Step 500-1200 | Rising | 0.01-0.10 | Medium | ⚠️ Watch |
| Step 1200+ | Diverging | 0.10-30.20 | **Very High** | ❌ Critical |

#### 🔍 Detailed Analysis

1. **Abnormal Divergence:**
   - Final KL: 9.37, far exceeding 0.05 warning threshold
   - Max KL: 30.20, critically high
   - **Interpretation:** Model severely deviates from base model
   - **Possible Causes:**
     - No SFT foundation, GRPO directly modifies base model
     - KL penalty beta=0.01 may be insufficient
     - Potential overfitting to training set

2. **Comparison with Experiment 5:**
   - Exp 5: Final KL 0.028, peak 0.055 (with SFT)
   - Exp 3-5: Final KL 9.37, peak 30.20 (no SFT)
   - **Conclusion:** SFT pretraining effectively constrains GRPO exploration

---

### 4️⃣ Policy Entropy

**Figure Location:** Subplot (d) - Bottom right

#### 📊 Data Summary

| Metric | Value | Level | Interpretation |
|------|--------|------|----------------|
| Initial Entropy | **0.46** | High | High randomness |
| Final Entropy | **3.43** | **Very High** | **Abnormally high** |
| Entropy Change | **+2.97** (+646%) | Abnormal | **Abnormal increase** |

#### 🔍 Detailed Analysis

1. **Abnormal Entropy Increase:**
   - Normal: Entropy should decrease (model becomes confident)
   - Actual: Entropy dramatically increases (model becomes random)
   - **Warning:** Training instability, potential model collapse

2. **Comparison with Experiment 5:**
   - Exp 5: Entropy 0.43 → 0.25 (normal)
   - Exp 3-5: Entropy 0.46 → 3.43 (abnormal)
   - **Conclusion:** Lack of SFT causes training instability

---

### 5️⃣ Gradient Norm

**Figure Location:** Subplot (e) - Bottom far right

#### ✅ Stability Assessment

| Metric | Value | Safe Threshold | Label | Status |
|------|--------|----------|----------------|--------|
| Max Grad Norm | **20.53** | < 1.0 | Warning | ❌ Critical |
| Mean Grad Norm | **2.39** | < 0.5 | Ideal | ⚠️ High |
| Explosion Risk | **High** | - | - | ⚠️ Warning |

#### 🔍 Detailed Analysis

1. **Gradient Instability:**
   - Max gradient norm 20.53, far exceeding 1.0 warning threshold
   - Mean gradient norm 2.39, 5× higher than Experiment 5
   - **Conclusion:** Unstable training, gradient explosion risk

2. **Comparison with Experiment 5:**
   - Exp 5: Max 0.69, Mean 0.45 (stable)
   - Exp 3-5: Max 20.53, Mean 2.39 (unstable)
   - **Reason:** Lack of SFT foundation causes optimization difficulties

---

## 🔄 Comparison with Experiment 5

### Key Differences Summary

| Metric | Experiment 5 (SFT+GRPO) | Experiment 3-5 (GRPO-only) | Difference |
|------|--------|-----------------|-------------------|----------|
| Initial Acc | 49.4% | 31.2% | SFT provides an 18.2% base |
| Final Acc | 71.2% | 50.0% | SFT gives 21.2% higher final performance |
| Final KL | 0.028 | 9.37 | No SFT causes severe deviation |
| Final Entropy | 0.25 ↓ | 3.43 ↑ | SFT keeps training stable |
| Max Grad | 0.69 | 20.53 | SFT lowers optimization difficulty |
| Stability | ✅ Stable | ❌ Unstable | SFT is key to stability |

### 💡 Key Insights

1. **Critical Role of SFT:**
   - Provides better initialization (49.4% vs 31.2%)
   - Constrains GRPO exploration (KL: 0.028 vs 9.37)
   - Ensures training stability (entropy ↓ vs ↑)
   - Improves final performance (71.2% vs 50.0%)

2. **Challenges of GRPO-only:**
   - Requires more steps to converge from scratch
   - Prone to model collapse (KL divergence, entropy growth)
   - Needs stronger regularization (larger beta)
   - Limited final performance

3. **Paper Recommendations:**
   - Strongly recommend two-stage SFT+GRPO training
   - Use GRPO-only as ablation to demonstrate SFT necessity
   - Clearly compare both methods in paper

---

## 🎨 Figure Caption Guide

### 📋 Recommended Caption (LaTeX Format)

#### English Version (For Paper)

```latex
\begin{figure*}[t]
    \centering
    \includegraphics[width=\textwidth]{exp3-5_training_visualization.png}
    \caption{Training dynamics of GRPO-only (without SFT pretraining) on GSM8K training set. 
    \textbf{(a)} In-training accuracy starts at 31.2\% and reaches 50.0\%, showing the challenge of training from base model without SFT foundation. Bold line represents smoothed trend; shaded area shows raw per-batch variance.
    \textbf{(b)} Training loss exhibits high volatility compared to SFT+GRPO approach.
    \textbf{(c)} KL divergence increases dramatically to 9.37 (peak: 30.20), far exceeding safe threshold, indicating severe deviation from base model.
    \textbf{(d)} Policy entropy abnormally increases from 0.46 to 3.43, suggesting training instability.
    \textbf{(e)} Gradient norms reach 20.53 (mean: 2.39), indicating optimization difficulties without SFT initialization.
    This experiment demonstrates the critical importance of SFT pretraining for stable GRPO training.}
    \label{fig:grpo_only_dynamics}
\end{figure*}
```

---

## 📝 Academic Writing Guidelines

### How to Use This Experiment in Paper

#### 1. As an Ablation Study

```
To demonstrate the importance of SFT pretraining, we conduct an ablation 
study by training GRPO directly on the base model (Experiment 3-5). 
Results show that without SFT initialization:
- Initial accuracy drops from 49.4% to 31.2%
- Final accuracy decreases from 71.2% to 50.0%
- KL divergence increases dramatically (0.028 → 9.37)
- Training becomes unstable (entropy increases instead of decreasing)

These findings confirm that SFT pretraining is essential for stable and 
effective GRPO training on mathematical reasoning tasks.
```

#### 2. Comparison Table

```latex
\begin{table}[t]
\centering
\caption{Comparison of SFT+GRPO vs GRPO-only}
\label{tab:sft_comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Metric} & \textbf{SFT+GRPO} & \textbf{GRPO-only} \\
\midrule
Initial Accuracy & 49.4\% & 31.2\% \\
Final Accuracy & \textbf{71.2\%} & 50.0\% \\
Peak Accuracy & 92.5\% & 100.0\%* \\
Final KL Divergence & 0.028 & 9.370 \\
Final Entropy & 0.25 & 3.43 \\
Max Gradient Norm & 0.69 & 20.53 \\
Training Stability & Stable & Unstable \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\item[*] Unstable, not sustained
\end{tablenotes}
\end{table}
```

---

## ❓ FAQ

### Q1: Why does GRPO-only perform so poorly?

**Core Reasons:**
1. **Lack of mathematical reasoning foundation:** Base model not finetuned on math tasks
2. **Excessive exploration space:** GRPO explores from scratch, easily deviates
3. **Training instability:** Without SFT constraints, model prone to collapse

---

### Q2: What is the value of this experiment?

**Academic Value:**
- ✅ Demonstrates necessity of SFT pretraining (ablation)
- ✅ Shows limitations of GRPO (needs good initialization)
- ✅ Provides rationale for two-stage training

**Practical Value:**
- ⚠️ Not recommended for real applications
- ✅ Can guide hyperparameter tuning (e.g., increase beta)

---

## ✅ Key Conclusions

1. **SFT pretraining is essential:** Experiment 3-5 shows direct GRPO training is poor and unstable
2. **Two-stage training superior:** Experiment 5 (SFT+GRPO) clearly outperforms 3-5 on all metrics
3. **Training stability critical:** KL divergence, entropy, gradient norms all show GRPO-only instability
4. **Paper recommendation:** Use this as ablation study to justify method design

---

**Document version:** v1.0  
**Last updated:** November 27, 2024  
**Authors:** CS566 Group Project — Experiment 3-5 team

---

**🎓 Compared with Experiment 5, this shows the value of SFT.**
