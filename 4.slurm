#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# GRPO model inference job configuration (for the ankner/gsm8k-CoT dataset)
# ============================================================================
# Dataset        | Samples | Time     | Memory | CPU | GPU    | Notes
# ---------------|---------|----------|--------|-----|--------|------------------
# Quick test     | 10      | 00:15:00 | 12G    | 4   | A100×1 | Validate model loading
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
#   - Model (4-bit) + LoRA: 4.5GB
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

# ============================================================================
# Full inference configuration (1320-sample GRPO inference)
# ============================================================================
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
echo "=========================================="
echo "🚀 GRPO Model Inference Job Starting"
echo "=========================================="
echo "Job started at: `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes"
echo "Running on $SLURM_NPROCS processors"
echo "GPU allocation: $CUDA_VISIBLE_DEVICES"
echo "=========================================="
echo ""

# Move to the correct directory
cd /home1/yihelu/csci566/
echo "Current working directory: `pwd`"

# Load conda environment
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate torch-env

# ============================================================================
# Run GRPO model inference
# ============================================================================
#
# What the script does:
#   1. Load the GRPO-trained LoRA model (qwen_grpo_lora_multigpu)
#   2. Run inference on the test set (train[924:934] or train[924:1319])
#   3. Compute accuracy and save detailed results
#   4. Output directory: inference_results_grpo/
#
# Output files:
#   - inference_results.json      : inference result for every sample
#   - evaluation_metrics.json     : accuracy and other evaluation metrics
#   - wrong_cases.json            : error-case analysis
#
# Switching configuration:
#   - Current: 10 test samples, train[924:934]
#   - Full inference: change line 45 of the code to train[924:1319]
#   - Also switch the SLURM configuration to the option with more resources
#
echo "📊 Starting GRPO model inference..."
python 4-qwen-grpo-inference.py
#
# ============================================================================

# Check exit status
EXIT_CODE=$?
echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Inference completed successfully!"
    echo "📁 Results saved in: inference_results_grpo/"
else
    echo "❌ Inference failed with exit code: $EXIT_CODE"
fi
echo "Job finished at: `date`"
echo "=========================================="
echo ""

exit $EXIT_CODE
