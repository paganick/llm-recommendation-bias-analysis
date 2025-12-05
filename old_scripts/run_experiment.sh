#!/bin/bash
#
# LLM Recommendation Bias Analysis - Batch Experiment Script
#
# Usage:
#   ./run_experiment.sh [OPTIONS]
#
# Or submit to cluster:
#   sbatch run_experiment.sh
#
# To configure SLURM, uncomment and modify the following lines:
# #SBATCH --job-name=llm_bias_analysis
# #SBATCH --output=logs/experiment_%j.log
# #SBATCH --error=logs/experiment_%j.err
# #SBATCH --time=02:00:00
# #SBATCH --mem=8G
# #SBATCH --cpus-per-task=4

# Exit on error
set -e

# Configuration - modify these as needed
PROVIDER="${PROVIDER:-openai}"
MODEL="${MODEL:-gpt-4o-mini}"
DATASET="${DATASET:-twitteraae}"
DATASET_VERSION="${DATASET_VERSION:-all_aa}"
DATASET_SIZE="${DATASET_SIZE:-1500}"
POOL_SIZE="${POOL_SIZE:-75}"
K="${K:-15}"
PROMPT_STYLE="${PROMPT_STYLE:-popular}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs/experiments}"

# Create log directory if using SLURM
mkdir -p logs

# Print configuration
echo "========================================================================"
echo "LLM RECOMMENDATION BIAS ANALYSIS"
echo "========================================================================"
echo "Configuration:"
echo "  Provider: $PROVIDER"
echo "  Model: $MODEL"
echo "  Dataset: $DATASET ($DATASET_VERSION)"
echo "  Dataset size: $DATASET_SIZE"
echo "  Pool size: $POOL_SIZE"
echo "  Recommendations (k): $K"
echo "  Prompt style: $PROMPT_STYLE"
echo "  Output directory: $OUTPUT_DIR"
echo "========================================================================"
echo ""

# Check API key
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: OPENAI_API_KEY not set!"
        echo "Set it with: export OPENAI_API_KEY='your-key-here'"
        exit 1
    fi
    echo "✓ OpenAI API key found"
elif [ "$PROVIDER" = "anthropic" ]; then
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "ERROR: ANTHROPIC_API_KEY not set!"
        echo "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
        exit 1
    fi
    echo "✓ Anthropic API key found"
fi

echo ""

# Run analysis
echo "Starting analysis..."
echo ""

python run_complete_analysis.py \
    --provider "$PROVIDER" \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --dataset-version "$DATASET_VERSION" \
    --dataset-size "$DATASET_SIZE" \
    --pool-size "$POOL_SIZE" \
    --k "$K" \
    --prompt-style "$PROMPT_STYLE" \
    --output-dir "$OUTPUT_DIR" \
    --include-gender \
    --include-political

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ ANALYSIS COMPLETED SUCCESSFULLY"
    echo "========================================================================"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "❌ ANALYSIS FAILED (exit code: $EXIT_CODE)"
    echo "========================================================================"
    echo ""
    exit $EXIT_CODE
fi
