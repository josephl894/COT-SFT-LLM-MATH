# GRPO Training Hyperparameters Summary

This document summarizes the hyperparameters used for training the Qwen-2.5-Coder-1.5B model using GRPO (Group Relative Policy Optimization).

## 1. Model Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-Coder-1.5B` | The pre-trained model used as a starting point. |
| **Quantization** | 4-bit (NF4) | Loaded in 4-bit using NF4 quantization to save memory. |
| **Compute Dtype** | `torch.float16` | Computation is performed in float16. |
| **Double Quant** | `True` | Enabled for further memory savings. |

## 2. LoRA Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Rank (r)** | 16 | The rank of the low-rank matrices. |
| **Alpha** | 32 | Scaling factor for LoRA updates (usually 2x rank). |
| **Target Modules** | All Linear Layers | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`. |
| **Dropout** | 0.05 | Dropout probability for LoRA layers. |
| **Bias** | "none" | No bias terms are trained. |
| **Task Type** | `CAUSAL_LM` | Causal Language Modeling. |

## 3. Training Configuration

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Epochs** | 1 | Number of training epochs. |
| **Batch Size (Per Device)** | 2 | Batch size per GPU. |
| **Gradient Accumulation** | 2 | Number of steps to accumulate gradients before update. |
| **Total Batch Size** | 8 (for 2 GPUs) | `2 (per_device) * 2 (accum) * 2 (GPUs)`. |
| **Gradient Checkpointing** | `False` | Disabled due to compatibility issues with DDP + LoRA. |
| **FP16** | `True` | Mixed precision training. |

## 4. Generation Configuration

These parameters control how the model generates responses during the exploration phase of GRPO.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Num Generations** | 4 | Number of candidate responses generated per prompt. |
| **Generation Batch Size** | 4 | Batch size for generation. |
| **Max Completion Length** | 256 | Maximum tokens for the generated response. |
| **Max Prompt Length** | 1024 | Maximum tokens for the input prompt. |
| **Temperature** | 0.9 | Controls randomness (higher = more random). |
| **Top P** | 0.9 | Nucleus sampling probability. |
| **Top K** | 50 | Top-K sampling. |

## 5. Optimizer & Scheduler

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | `paged_adamw_8bit` | Memory-efficient optimizer. |
| **Learning Rate** | 5e-5 | Initial learning rate. |
| **Warmup Ratio** | 0.1 (10%) | Percentage of steps for learning rate warmup. |
| **Max Grad Norm** | 1.0 | Gradient clipping threshold. |

## 6. GRPO Specifics

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Beta** (KL Penalty) | 0.01 | KL divergence penalty coefficient. Keeps the model close to the reference. |
| **Reward Function** | Outcome-based | **1.0** for correct answer, **0.0** for incorrect. |

## 7. Hardware & Environment

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Multi-GPU Strategy** | DDP (Distributed Data Parallel) | Managed via `accelerate`. |
| **Device Map** | `{'': LOCAL_RANK}` | Ensures correct model placement on GPUs. |
