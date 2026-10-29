#!/bin/bash
#SBATCH --account=liu32_1378
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1

# ============================================================================
# Test job configuration
# ============================================================================
# Purpose: test the answer_utils.py module
# Data: load only 5 samples for validation
# Expected runtime: < 5 minutes
#
# What the job does:
# 1. Load the first 5 rows of the GSM8K-CoT dataset
# 2. Show the raw data structure
# 3. Walk through each step of answer extraction
# 4. Check that the comparison logic is correct
# 5. Simulate the accuracy calculation
# ============================================================================

# ============================================================================
# Resources (test job — minimum configuration)
# ============================================================================
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a40|a100"

# ============================================================================
# Email notification
# ============================================================================
#SBATCH --mail-user=yihelu@usc.edu
#SBATCH --mail-type=END,FAIL

# ============================================================================
# GPU configuration (the test job does not need a specific GPU; any available GPU is fine)
# ============================================================================
# The test mainly checks data-processing logic, so GPU performance does not matter much
# Leaving the GPU model unspecified gets a resource sooner
#
# To pin a specific GPU (not recommended for this test):
# #SBATCH --gpus-per-task=v100:1
# #SBATCH --gpus-per-task=a40:1
# ============================================================================

echo ""
echo "=========================================="
echo "  Answer Utils module test"
echo "=========================================="
echo ""
echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo ""

# Move to the correct directory
cd /home1/yihelu/csci566/
echo "Current working directory is `pwd`"
echo ""

# ============================================================================
# Environment
# ============================================================================
echo "🔧 Configuring environment..."
source /apps/conda/miniforge3/24.3.0/etc/profile.d/conda.sh

echo "📦 Activating torch-env conda environment..."
conda activate torch-env

# Check the environment
echo ""
echo "🔍 Environment verification:"
python --version
echo "Python path: $(which python)"
echo ""

# ============================================================================
# Run the test
# ============================================================================
echo "=========================================="
echo "  Starting the test script"
echo "=========================================="
echo ""

python test_answer_utils.py

# Save the exit code
TEST_EXIT_CODE=$?

echo ""
echo "=========================================="
echo "  Test finished"
echo "=========================================="
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo "✅ Test finished successfully!"
else
    echo "❌ Test failed, exit code: $TEST_EXIT_CODE"
fi

echo ""
echo "Program finished at: `date`"
echo ""

# Exit with the test script's exit code
exit $TEST_EXIT_CODE
