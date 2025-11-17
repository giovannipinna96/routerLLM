#!/bin/bash
# Quick Start Script for Enhanced RouterLLM System
# This script runs the corrected system with 100B+ model support and dynamic routing

echo "=========================================="
echo "RouterLLM Enhanced System - Quick Start"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "Python version: $python_version"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python3 -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}'); [print(f'  GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]" 2>/dev/null || echo "No GPUs detected"

# Create directories
echo ""
echo "Setting up directories..."
mkdir -p results/production
mkdir -p logs/carbon
mkdir -p models
mkdir -p data

# Install dependencies if needed
echo ""
echo "Checking dependencies..."
pip install -q datasets transformers torch accelerate bitsandbytes codecarbon 2>/dev/null

# Select test mode
echo ""
echo "Select test mode:"
echo "1) Quick Test (10 examples, complexity router, smaller models)"
echo "2) Standard Test (50 examples, dynamic router, 70B model)"
echo "3) Full Test (50 examples, dynamic router, 100B+ model) [Requires 2-4 A100 GPUs]"
echo "4) Generate training data only"
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Running Quick Test..."
        python3 scripts/humaneval_comparison.py \
            --config configs/default_config.yaml \
            --num-examples 10 \
            --seed 42 \
            --results-dir results/quick_test
        ;;
    2)
        echo ""
        echo "Running Standard Test with Dynamic Router..."
        python3 scripts/enhanced_humaneval_comparison.py \
            --config configs/production_config.yaml \
            --num-examples 50 \
            --use-dynamic-router \
            --seed 42 \
            --results-dir results/standard_test
        ;;
    3)
        echo ""
        echo "Running Full Test with 100B+ Model..."
        echo "WARNING: This requires 200GB+ GPU memory (2-4 A100 80GB GPUs)"
        read -p "Continue? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            python3 scripts/enhanced_humaneval_comparison.py \
                --config configs/production_config.yaml \
                --num-examples 50 \
                --use-dynamic-router \
                --use-large-model \
                --seed 42 \
                --results-dir results/full_test
        else
            echo "Test cancelled."
        fi
        ;;
    4)
        echo ""
        echo "Generating training data..."
        python3 main.py generate-data --samples 1200 --output-dir ./data
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Test completed! Check results in the results directory."
echo "=========================================="

# Display results if available
if [ -f "results/production/comparison_analysis.json" ]; then
    echo ""
    echo "Summary of results:"
    python3 -c "
import json
with open('results/production/comparison_analysis.json') as f:
    data = json.load(f)
    print(f\"Router Accuracy: {data['accuracy_comparison']['router_system']['accuracy']:.1%}\")
    print(f\"Direct LLM Accuracy: {data['accuracy_comparison']['direct_large_llm']['accuracy']:.1%}\")
    print(f\"Carbon Reduction: {data['carbon_footprint_comparison']['carbon_reduction_percentage']:.1f}%\")
    print(f\"Recommendation: {data['conclusion']['overall_recommendation']}\")
" 2>/dev/null || echo "Could not parse results file."
fi

echo ""
echo "For detailed analysis, check:"
echo "  - results/*/comparison_analysis.json - Full comparison data"
echo "  - results/*/router_results.json - Router system details"
echo "  - results/*/direct_results.json - Direct LLM details"
echo "  - logs/ - System and carbon tracking logs"
