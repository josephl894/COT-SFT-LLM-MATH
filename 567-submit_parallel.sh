#!/bin/bash
# ============================================================================
# Parallel job submission script
# Purpose: submit several training + evaluation jobs at once; output directories are isolated by Job ID
# ============================================================================

echo "🚀 Submitting SFT+GRPO training and evaluation jobs..."
echo ""

# 1. Submit the training job and capture the Job ID
JOB5=$(sbatch --parsable 5.slurm)

if [ -z "$JOB5" ]; then
    echo "❌ Error: training job submission failed"
    exit 1
fi

echo "✅ Training job submitted"
echo "   Job ID: $JOB5"
echo "   Output directory: qwen_sft_grpo_lora_${JOB5}/"
echo ""

# 2. Submit the full evaluation job (after training succeeds)
echo "📝 Submitting full-model evaluation (6.slurm)..."
JOB6=$(sbatch --parsable --dependency=afterok:$JOB5 --export=ALL,TRAINING_JOB_ID=$JOB5 6.slurm)

if [ -z "$JOB6" ]; then
    echo "❌ Error: evaluation job 6 submission failed"
else
    echo "✅ Full evaluation job submitted"
    echo "   Job ID: $JOB6"
    echo "   Dependency: runs after training job $JOB5 succeeds"
    echo "   Output directory: inference_results_sft_grpo_${JOB5}/"
fi
echo ""

# 3. Submit the ablation evaluation job (after training succeeds)
echo "📝 Submitting ablation evaluation (7.slurm)..."
JOB7=$(sbatch --parsable --dependency=afterok:$JOB5 --export=ALL,TRAINING_JOB_ID=$JOB5 7.slurm)

if [ -z "$JOB7" ]; then
    echo "❌ Error: evaluation job 7 submission failed"
else
    echo "✅ Ablation evaluation job submitted"
    echo "   Job ID: $JOB7"
    echo "   Dependency: runs after training job $JOB5 succeeds"
    echo "   Output directory: inference_results_grpo_only_${JOB5}/"
fi
echo ""

# 4. Summary
echo "=========================================="
echo "📊 Submission summary"
echo "=========================================="
echo "Training job: $JOB5 (qwen_sft_grpo_lora_${JOB5}/)"
echo "Full evaluation: $JOB6 (waits for $JOB5)"
echo "Ablation evaluation: $JOB7 (waits for $JOB5)"
echo "=========================================="
echo ""
echo "💡 Monitor:"
echo "   Queue: squeue -u \$USER"
echo "   Log: tail -f slurm-${JOB5}-*.out"
echo ""
echo "💡 To submit more parallel jobs, run this script again"
