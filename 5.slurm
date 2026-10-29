#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:2
#SBATCH --constraint="v100|a40|a100"
#SBATCH --output=slurm-%j-sft-grpo-train.out
#SBATCH --mail-user=your_email@example.com
#SBATCH --mail-type=END,FAIL

# ============================================================================
# SFT + GRPO training job (7470 samples)
# Multi-GPU mode (2 GPUs, gradient_checkpointing=False)
# Estimated time: ~3-4 hours | safety margin: 12× | configured time: 48 hours
# ============================================================================

source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh
conda activate torch-env

echo "Starting SFT+GRPO Multi-GPU Training Job..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Date: `date`"

# Launch multi-GPU training with accelerate
accelerate launch --config_file accelerate_config_2gpu.yaml 5-qwen-sft-grpo-train.py

echo "Done at `date`"
