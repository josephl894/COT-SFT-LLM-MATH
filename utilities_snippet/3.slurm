#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Recommended resources (choose by dataset size)
# ============================================================================
# Dataset size | Time      | Memory | CPU | GPU | Notes
# -------------|-----------|--------|-----|-----|------------------
# 10 samples   | 00:15:00  | 16G    | 4   | 1   | Quick test
# 100 samples  | 00:30:00  | 32G    | 8   | 1   | Small-scale training (current configuration)
# 1000 samples | 02:00:00  | 32G    | 8   | 1   | Medium scale
# Full dataset | 08:00:00  | 48G    | 16  | 1   | Full training (7.5k samples)
# ============================================================================

# ============================================================================
# GPU policy: no model specified = accept any available GPU (L40S/A100/A40/V100)
# Benefit: much faster to get a resource, less time in the queue
# Note: the system may assign a P100 (CUDA 6.0); the code detects that and prints a warning
# ============================================================================
# Available GPU models (sorted by performance):
#   l40s (CUDA 8.9) > a100 (CUDA 8.0) > a40 (CUDA 8.6) > v100 (CUDA 7.0)
#   ⚠️ p100 (CUDA 6.0) is not compatible with PyTorch 2.8+
# ============================================================================



# Quick validation configuration (10-sample GRPO training)
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a40|a100"

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

python 3-qwen-grpo-train.py

# Run Jupyter Notebook with papermill
# papermill HW2_microsoft_malware_prediction__yihelu.ipynb output_notebook.ipynb

# End of the program
echo ""
echo "Program finished with exit code $? at: `date`"
echo ""
