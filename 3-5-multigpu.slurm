#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Multi-GPU training configuration (ankner/gsm8k-CoT dataset, GRPO training)
# ============================================================================
# Dataset       | GPUs | Time     | Memory | CPU | Notes
# --------------|------|----------|--------|-----|------------------
# Test          | 2    | 00:15:00 | 32G    | 12  | Quick test (10 samples)
# Small scale   | 2    | 00:30:00 | 40G    | 12  | Development debugging (100 samples)
# Medium scale  | 2    | 01:30:00 | 48G    | 16  | Medium scale (1000 samples)
# Full training | 2    | 06:00:00 | 64G    | 16  | Full training (7470 samples, current configuration)
# ============================================================================
#
# Calculation (full GRPO training, 2×A100):
#   - Effective batch size: 2 (per_device) × 2 (grad_accum) × 2 (GPUs) = 8
#   - Total steps: 7470 / 8 = 934 steps
#   - Estimated time: 934 × 8s = 7472s ≈ 2.08 hours (GRPO must generate multiple candidate responses)
#   - Load + save overhead: +30 minutes
#   - DDP sync overhead: +15%
#   - Safety margin (×2): 6 hours
#
# VRAM use (each A100 40GB):
#   - Model (4-bit): 4GB
#   - LoRA + optimizer + activations: 8-10GB
#   - GRPO generation cache: 3-4GB
#   - Headroom: ~23GB, sufficient
#
# Multi-GPU speedup:
#   - Theoretical speedup: 2×
#   - Actual speedup: ~1.7-1.8× (communication overhead)
# ============================================================================

# Current configuration: full training (7470 samples, 2×GPU)
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="v100|a40|a100"
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=END,FAIL


# ============================================================================
# Multi-GPU strategy:
# 1. PyTorch DDP (DistributedDataParallel): enabled automatically, no manual setup
# 2. accelerate detects SLURM environment variables and initializes multi-GPU
# 3. Each GPU runs the same code but processes a different batch
# 4. Gradients are synchronized across GPUs automatically
# ============================================================================

echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"

# Move to the correct directory
cd /home1/yihelu/csci566/
echo "Current working directory is `pwd`"

# Load conda environment
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate torch-env

# ============================================================================
# Multi-GPU DDP training configuration
# ============================================================================
#
# ⚠️ Important: DDP + gradient_checkpointing + LoRA has a known conflict
# Error: RuntimeError: Expected to mark a variable ready only once
# Fix: gradient_checkpointing must be False (already set in the code)
#
# DDP mode:
#   ✓ Simple configuration, no extra dependencies
#   ✓ Faster training (1.8× a single GPU)
#   ✓ Fully compatible with 4-bit quantization + LoRA
#   ✗ Cannot use gradient_checkpointing
#   📊 VRAM use: ~14GB/GPU (Qwen2.5-1.5B)
#
# If VRAM is insufficient, you can:
#   - Decrease per_device_train_batch_size (set to 1 in the code)
#   - Increase gradient_accumulation_steps (set to 4 in the code)
#   - Decrease max_completion_length (set to 128 in the code)
#
echo "🚀 Launching 2-GPU DDP training..."
accelerate launch --config_file accelerate_config_2gpu.yaml 3-5-qwen-grpo-train-multigpu.py
#
# ============================================================================

echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
