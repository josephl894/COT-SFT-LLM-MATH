#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Recommended resources (for the ankner/gsm8k-CoT dataset)
# ============================================================================
# Dataset         | Samples | Time     | Memory | CPU | GPU    | Notes
# ----------------|---------|----------|--------|-----|--------|------------------
# Test            | 10      | 00:15:00 | 12G    | 4   | A100×1 | Quick check
# Full inference  | 1320    | 03:00:00 | 20G    | 8   | A100×1 | Full test set (current configuration)
# ============================================================================
#
# Calculation (full inference):
#   - Test-set samples: 1320
#   - Time per sample: ~5s (generate 256 tokens @ 50 tokens/s)
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



## Full inference configuration (1320-sample inference)
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

# Move to the correct directory
cd /home1/yihelu/csci566/
echo "Current working directory is `pwd`"

# Load Python module if required
# module load python  # Adjust based on your system if necessary
# module load conda

# conda init
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh

conda activate torch-env


python 2-qwen-cot-inference.py

# Run Jupyter Notebook with papermill
# papermill HW2_microsoft_malware_prediction__yihelu.ipynb output_notebook.ipynb

# End of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
