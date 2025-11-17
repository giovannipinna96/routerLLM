#!/bin/bash
#SBATCH --job-name=routerllm_pipeline
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --gpus=1
#SBATCH --partition=lovelace

################################################################################
# RouterLLM Full Pipeline SLURM Script
#
# This script runs the complete RouterLLM pipeline:
# 1. Generate synthetic dataset
# 2. Load HuggingFace tiny-codes dataset
# 3. Train BERT router on synthetic data
# 4. Train BERT router on tiny-codes data
# 5. Test all routers (BERT x2, Dummy, Graham) with carbon tracking
# 6. Aggregate results to JSON and CSV
#
# Usage:
#   sbatch scripts/run_full_pipeline.slurm          # Uses 50k samples (default)
#   sbatch --export=DATASET_SIZE=full scripts/run_full_pipeline.slurm  # Uses full 1.6M samples
#
################################################################################

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

################################################################################
# Configuration
################################################################################

# Dataset size: "50k" or "full" (1.6M)
DATASET_SIZE=${DATASET_SIZE:-"50k"}

# Training options: set USE_II_LOSS=true to use Inter-Intra Loss
USE_II_LOSS=${USE_II_LOSS:-false}

# Set SKIP_STANDARD_TRAINING=true to skip CrossEntropy training (if already done)
SKIP_STANDARD_TRAINING=${SKIP_STANDARD_TRAINING:-false}

# Project directories
PROJECT_DIR="/u/gpinna/phd_projects/routerLLM/routerLLM"
DATA_DIR="${PROJECT_DIR}/data"
MODELS_DIR="${PROJECT_DIR}/models"
RESULTS_DIR="${PROJECT_DIR}/results"
LOGS_DIR="${PROJECT_DIR}/logs"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"

# Dataset paths
SYNTHETIC_DIR="${DATA_DIR}/synthetic"
TINYCODES_DIR="${DATA_DIR}/tinycodes"

# Model paths - use different directories for II-loss models
if [ "${USE_II_LOSS}" == "true" ]; then
    BERT_SYNTHETIC_DIR="${MODELS_DIR}/bert_synthetic_iiloss"
    BERT_TINYCODES_DIR="${MODELS_DIR}/bert_tinycodes_iiloss"
else
    BERT_SYNTHETIC_DIR="${MODELS_DIR}/bert_synthetic"
    BERT_TINYCODES_DIR="${MODELS_DIR}/bert_tinycodes"
fi

# Configuration based on dataset size
if [ "$DATASET_SIZE" == "50k" ]; then
    SYNTHETIC_SAMPLES=10000
    TINYCODES_SAMPLES=50000
    EPOCHS=1
    BATCH_SIZE=16
elif [ "$DATASET_SIZE" == "full" ]; then
    SYNTHETIC_SAMPLES=50000
    TINYCODES_SAMPLES=0  # 0 means load all samples
    EPOCHS=3
    BATCH_SIZE=16
else
    echo "ERROR: Invalid DATASET_SIZE='${DATASET_SIZE}'. Must be '50k' or 'full'"
    exit 1
fi

################################################################################
# Setup
################################################################################

echo "============================================================"
echo "RouterLLM Full Pipeline"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Start time: $(date)"
echo "Node: ${SLURMD_NODENAME}"
echo "Dataset size: ${DATASET_SIZE}"
echo "Synthetic samples: ${SYNTHETIC_SAMPLES}"
echo "Tiny-codes samples: ${TINYCODES_SAMPLES}"
echo "Use Inter-Intra Loss: ${USE_II_LOSS}"
echo "============================================================"
echo ""

# Change to project directory
cd "${PROJECT_DIR}"

# Create necessary directories
mkdir -p "${SYNTHETIC_DIR}" "${TINYCODES_DIR}"
mkdir -p "${BERT_SYNTHETIC_DIR}" "${BERT_TINYCODES_DIR}"
mkdir -p "${RESULTS_DIR}" "${LOGS_DIR}/carbon"

# Load environment (if using modules)
# module load cuda/11.8 python/3.10  # Uncomment and adjust as needed

# Check GPU availability
echo "Checking GPU availability..."
nvidia-smi || echo "WARNING: nvidia-smi failed. GPU may not be available."
echo ""

################################################################################
# STEP 1: Generate Synthetic Dataset
################################################################################

echo "============================================================"
echo "STEP 1: Generating Synthetic Dataset"
echo "============================================================"
echo "Samples: ${SYNTHETIC_SAMPLES}"
echo "Output directory: ${SYNTHETIC_DIR}"
echo "Start time: $(date)"
echo ""

uv run python main.py generate-data \
    --samples ${SYNTHETIC_SAMPLES} \
    --output-dir "${SYNTHETIC_DIR}" \
    --seed 42

echo ""
echo "Synthetic dataset generation completed at $(date)"
echo "Generated files:"
ls -lh "${SYNTHETIC_DIR}"
echo ""

################################################################################
# STEP 2: Load HuggingFace Tiny-Codes Dataset
################################################################################

echo "============================================================"
echo "STEP 2: Loading HuggingFace Tiny-Codes Dataset"
echo "============================================================"
echo "Dataset: nampdn-ai/tiny-codes"
echo "Max samples: ${TINYCODES_SAMPLES} (0 = all)"
echo "Output directory: ${TINYCODES_DIR}"
echo "Start time: $(date)"
echo ""

if [ "$TINYCODES_SAMPLES" -eq 0 ]; then
    # Load full dataset
    uv run python main.py generate-data-hf \
        --dataset-name nampdn-ai/tiny-codes \
        --output-dir "${TINYCODES_DIR}" \
        --train-size 0.8 \
        --val-size 0.1 \
        --test-size 0.1 \
        --balance-classes \
        --seed 42
else
    # Load subset
    uv run python main.py generate-data-hf \
        --dataset-name nampdn-ai/tiny-codes \
        --output-dir "${TINYCODES_DIR}" \
        --max-samples ${TINYCODES_SAMPLES} \
        --train-size 0.8 \
        --val-size 0.1 \
        --test-size 0.1 \
        --balance-classes \
        --seed 42
fi

echo ""
echo "Tiny-codes dataset loading completed at $(date)"
echo "Generated files:"
ls -lh "${TINYCODES_DIR}"
echo ""

################################################################################
# STEP 3: Train BERT Router on Synthetic Dataset
################################################################################

# Skip if SKIP_STANDARD_TRAINING=true AND not using II-loss
if [ "${SKIP_STANDARD_TRAINING}" == "true" ] && [ "${USE_II_LOSS}" == "false" ]; then
    echo "============================================================"
    echo "STEP 3: SKIPPED (Standard Training)"
    echo "============================================================"
    echo "Reason: SKIP_STANDARD_TRAINING=true"
    echo "Using existing models from: ${BERT_SYNTHETIC_DIR}"
    echo ""
else
    echo "============================================================"
    echo "STEP 3: Training BERT Router (Synthetic Dataset)"
    echo "============================================================"
    echo "Data directory: ${SYNTHETIC_DIR}"
    echo "Model directory: ${BERT_SYNTHETIC_DIR}"
    echo "Epochs: ${EPOCHS}"
    echo "Batch size: ${BATCH_SIZE}"
    echo "Start time: $(date)"
    echo ""

    # Build training command with optional II-loss flag
    TRAIN_CMD="uv run python main.py train \
        --data-dir \"${SYNTHETIC_DIR}\" \
        --model-dir \"${BERT_SYNTHETIC_DIR}\" \
        --model-name bert-base-uncased \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate 2e-5"

    if [ "${USE_II_LOSS}" == "true" ]; then
        TRAIN_CMD="${TRAIN_CMD} --use-inter-intra-loss"
        echo "Using Inter-Intra Loss"
    fi

    eval $TRAIN_CMD
fi

echo ""
echo "BERT (synthetic) training completed at $(date)"
echo "Trained models:"
ls -lh "${BERT_SYNTHETIC_DIR}"
echo ""

################################################################################
# STEP 4: Train BERT Router on Tiny-Codes Dataset
################################################################################

# Skip if SKIP_STANDARD_TRAINING=true AND not using II-loss
if [ "${SKIP_STANDARD_TRAINING}" == "true" ] && [ "${USE_II_LOSS}" == "false" ]; then
    echo "============================================================"
    echo "STEP 4: SKIPPED (Standard Training)"
    echo "============================================================"
    echo "Reason: SKIP_STANDARD_TRAINING=true"
    echo "Using existing models from: ${BERT_TINYCODES_DIR}"
    echo ""
else
    echo "============================================================"
    echo "STEP 4: Training BERT Router (Tiny-Codes Dataset)"
    echo "============================================================"
    echo "Data directory: ${TINYCODES_DIR}"
    echo "Model directory: ${BERT_TINYCODES_DIR}"
    echo "Epochs: ${EPOCHS}"
    echo "Batch size: ${BATCH_SIZE}"
    echo "Start time: $(date)"
    echo ""

    # Build training command with optional II-loss flag
    TRAIN_CMD="uv run python main.py train \
        --data-dir \"${TINYCODES_DIR}\" \
        --model-dir \"${BERT_TINYCODES_DIR}\" \
        --model-name bert-base-uncased \
        --epochs ${EPOCHS} \
        --batch-size ${BATCH_SIZE} \
        --learning-rate 2e-5"

    if [ "${USE_II_LOSS}" == "true" ]; then
        TRAIN_CMD="${TRAIN_CMD} --use-inter-intra-loss"
        echo "Using Inter-Intra Loss"
    fi

    eval $TRAIN_CMD
fi

echo ""
echo "BERT (tiny-codes) training completed at $(date)"
echo "Trained models:"
ls -lh "${BERT_TINYCODES_DIR}"
echo ""

################################################################################
# STEP 5: Test All Routers with Carbon Tracking
################################################################################

echo "============================================================"
echo "STEP 5: Testing All Routers"
echo "============================================================"
echo "Routers to test:"
echo "  1. BERT (trained on synthetic data)"
echo "  2. BERT (trained on tiny-codes data)"
echo "  3. Dummy (random selection baseline)"
echo "  4. Graham Complexity (pre-trained)"
echo ""
echo "Start time: $(date)"
echo ""

# Define test prompts (can be expanded)
TEST_EXAMPLES_FLAG="--test-examples"

# Find the best BERT models
BERT_SYNTHETIC_MODEL=$(find "${BERT_SYNTHETIC_DIR}" -name "best_router*.pt" | head -n 1)
BERT_TINYCODES_MODEL=$(find "${BERT_TINYCODES_DIR}" -name "best_router*.pt" | head -n 1)

# If best_router*.pt not found, try router_epoch*.pt
if [ -z "$BERT_SYNTHETIC_MODEL" ]; then
    BERT_SYNTHETIC_MODEL=$(find "${BERT_SYNTHETIC_DIR}" -name "router_epoch*.pt" | tail -n 1)
fi

if [ -z "$BERT_TINYCODES_MODEL" ]; then
    BERT_TINYCODES_MODEL=$(find "${BERT_TINYCODES_DIR}" -name "router_epoch*.pt" | tail -n 1)
fi

echo "BERT Synthetic Model: ${BERT_SYNTHETIC_MODEL}"
echo "BERT Tiny-codes Model: ${BERT_TINYCODES_MODEL}"
echo ""

# Test 1: BERT (Synthetic)
echo "------------------------------------------------------------"
echo "Testing BERT Router (Synthetic Dataset)"
echo "------------------------------------------------------------"
uv run python "${SCRIPTS_DIR}/test_router_and_save.py" \
    --router-type bert \
    --router-model "${BERT_SYNTHETIC_MODEL}" \
    --output-file "${RESULTS_DIR}/bert_synthetic_results.json" \
    --router-name "BERT_Synthetic" \
    ${TEST_EXAMPLES_FLAG}
echo ""

# Test 2: BERT (Tiny-codes)
echo "------------------------------------------------------------"
echo "Testing BERT Router (Tiny-Codes Dataset)"
echo "------------------------------------------------------------"
uv run python "${SCRIPTS_DIR}/test_router_and_save.py" \
    --router-type bert \
    --router-model "${BERT_TINYCODES_MODEL}" \
    --output-file "${RESULTS_DIR}/bert_tinycodes_results.json" \
    --router-name "BERT_TinyCodes" \
    ${TEST_EXAMPLES_FLAG}
echo ""

# Test 3: Dummy Router
echo "------------------------------------------------------------"
echo "Testing Dummy Router (Baseline)"
echo "------------------------------------------------------------"
uv run python "${SCRIPTS_DIR}/test_router_and_save.py" \
    --router-type dummy \
    --output-file "${RESULTS_DIR}/dummy_results.json" \
    --router-name "Dummy" \
    ${TEST_EXAMPLES_FLAG}
echo ""

# Test 4: Graham Complexity Router
echo "------------------------------------------------------------"
echo "Testing Graham Complexity Router (Pre-trained)"
echo "------------------------------------------------------------"
uv run python "${SCRIPTS_DIR}/test_router_and_save.py" \
    --router-type graham_complexity \
    --output-file "${RESULTS_DIR}/graham_results.json" \
    --router-name "Graham_Complexity" \
    ${TEST_EXAMPLES_FLAG}
echo ""

echo "All router testing completed at $(date)"
echo ""

################################################################################
# STEP 6: Aggregate Results
################################################################################

echo "============================================================"
echo "STEP 6: Aggregating Results"
echo "============================================================"
echo "Input: ${RESULTS_DIR}/*_results.json"
echo "Output CSV: ${RESULTS_DIR}/router_comparison.csv"
echo "Start time: $(date)"
echo ""

uv run python "${SCRIPTS_DIR}/aggregate_results.py" \
    --results-dir "${RESULTS_DIR}" \
    --output-csv "${RESULTS_DIR}/router_comparison.csv"

echo ""
echo "Results aggregation completed at $(date)"
echo ""

################################################################################
# Summary
################################################################################

echo "============================================================"
echo "Pipeline Completed Successfully"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Generated Files:"
echo "------------------------------------------------------------"
echo "Datasets:"
echo "  - ${SYNTHETIC_DIR}/train_dataset.json"
echo "  - ${TINYCODES_DIR}/train_dataset.json"
echo ""
echo "Trained Models:"
echo "  - ${BERT_SYNTHETIC_MODEL}"
echo "  - ${BERT_TINYCODES_MODEL}"
echo ""
echo "Results:"
ls -lh "${RESULTS_DIR}/"*.json 2>/dev/null || echo "  No JSON results found"
echo "  - ${RESULTS_DIR}/router_comparison.csv"
echo ""
echo "Carbon Footprint:"
ls -lh "${LOGS_DIR}/carbon/"*.csv 2>/dev/null || echo "  No carbon logs found"
echo ""
echo "Logs:"
echo "  - ${LOGS_DIR}/slurm_${SLURM_JOB_ID}.out"
echo "  - ${LOGS_DIR}/slurm_${SLURM_JOB_ID}.err"
echo "============================================================"

exit 0
