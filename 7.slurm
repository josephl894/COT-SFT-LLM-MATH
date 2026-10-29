#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a40|a100"
#SBATCH --output=slurm-%j-grpo-only-eval.out
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=END,FAIL

# ============================================================================
# GRPO-only ablation evaluation (full test set: 1319 samples)
# Estimated time: ~3 hours | safety margin: 8× | configured time: 24 hours
# ============================================================================

source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate torch-env

echo "🚀 Starting GRPO-only Evaluation (Ablation Study)..."
echo "⚠️  Testing GRPO LoRA without SFT foundation"
python 7-qwen-sftbased-grpo-only-eval.py
