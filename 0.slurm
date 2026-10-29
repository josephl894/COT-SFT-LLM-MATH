#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Baseline inference job configuration (for the ankner/gsm8k-CoT dataset)
# ============================================================================
# Dataset        | Samples | Time     | Memory | CPU | GPU    | Notes
# ---------------|---------|----------|--------|-----|--------|------------------
# Quick test     | 10      | 00:15:00 | 12G    | 4   | A100×1 | Validate the inference pipeline
# Full inference | 1320    | 03:00:00 | 20G    | 8   | A100×1 | Full test set (current configuration)
# ============================================================================
#
# Calculation (full inference):
#   - Test-set samples: 1320
#   - Time per sample: ~5s (generate 256 tokens @ 50 tokens/s on A100)
#   - Inference time: 1320 × 5s = 6600s ≈ 1.83 hours
#   - Load + save overhead: +20 minutes
#   - Safety margin (×1.5): 3 hours
#
# VRAM use (A100 40GB):
#   - Model (4-bit): 4GB
#   - KV cache: 2-3GB
#   - Headroom: ~33GB, ample
# ============================================================================

# ============================================================================
# GPU policy: request A100 for the best performance
# ============================================================================
# Available GPU models (sorted by performance):
#   l40s (CUDA 8.9) > a100 (CUDA 8.0) > a40 (CUDA 8.6) > v100 (CUDA 7.0)
#   ⚠️ p100 (CUDA 6.0) is not compatible with PyTorch 2.8+
#
# Inference job characteristics:
#   - GPU performance requirements are lower than for training
#   - A100: fastest inference (~50 tokens/s)
#   - A40/V100: also sufficient (~30-40 tokens/s)
#
# Flexible policy: if the queue is too long, change to --gpus-per-task=1 to accept any GPU
# ============================================================================

# Current configuration: full inference (1320-sample baseline inference)
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a40|a100"

# ============================================================================
# Email notification
# ============================================================================
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=END,FAIL

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
# Run baseline inference
# ============================================================================
#
# Task:
#   - Use the original Qwen2.5-Coder-1.5B model (no fine-tuning)
#   - Evaluate on test slice train[924:1319] (395 samples)
#   - Comparison baseline for the SFT and GRPO models
#
# Output directory: inference_results_baseline/
#   - inference_results.json: detailed inference results
#   - evaluation_metrics.json: evaluation metrics
#   - wrong_cases.json: error-case analysis
#
echo "🚀 Starting Baseline Model Inference..."
python 0-qwen-baseline-inference.py
#
# ============================================================================

# End of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
