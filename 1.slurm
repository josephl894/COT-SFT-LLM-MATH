#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Recommended resources (for the ankner/gsm8k-CoT dataset)
# ============================================================================
# Dataset        | Samples | Time     | Memory | CPU | GPU    | Notes
# ---------------|---------|----------|--------|-----|--------|------------------
# Test           | 10      | 00:15:00 | 16G    | 4   | A100×1 | Quick test
# Small scale    | 100     | 00:20:00 | 20G    | 8   | A100×1 | Development debugging
# Medium scale   | 1000    | 00:45:00 | 24G    | 8   | A100×1 | Medium scale
# Full training  | 7470    | 03:00:00 | 32G    | 12  | A100×1 | Full training (current configuration)
# ============================================================================
#
# Calculation (full training):
#   - Effective batch size: 1 × 4 = 4
#   - Total steps: 7470 / 4 = 1868 steps
#   - Estimated time: 1868 × 2s = 3736s ≈ 1 hour + 20 minutes overhead = 1.5 hours
#   - Safety margin (×2): 3 hours
#
# VRAM use (A100 40GB):
#   - Model (4-bit): 4GB
#   - LoRA + optimizer + activations: 8-10GB
#   - Headroom: ~26GB, sufficient
# ============================================================================

# ============================================================================
# GPU policy: request A100 for the best performance
# ============================================================================
# Available GPU models (sorted by performance):
#   l40s (CUDA 8.9) > a100 (CUDA 8.0) > a40 (CUDA 8.6) > v100 (CUDA 7.0)
#   ⚠️ p100 (CUDA 6.0) is not compatible with PyTorch 2.8+
#
# A100 advantages:
#   ✓ 40GB/80GB VRAM is sufficient
#   ✓ Training is fast (about 2× a V100)
#   ✓ Supports automatic TF32/BF16 acceleration
#   ⚠️ The queue may be long (USC CARC, a busy GPU)
#
# Flexible policy: if the queue is too long, change to --gpus-per-task=1 to accept any GPU
# ============================================================================



## Full training configuration (7470 samples)
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
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

python 1-qwen-cot-sft.py

# Run Jupyter Notebook with papermill
# papermill HW2_microsoft_malware_prediction__yihelu.ipynb output_notebook.ipynb

# End of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
