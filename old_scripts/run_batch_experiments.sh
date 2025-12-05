#!/bin/bash
#
# Run Multiple LLM Bias Analysis Experiments
#
# This script runs multiple experiments with different configurations
# to compare bias across different prompt styles, models, etc.
#

set -e

echo "========================================================================"
echo "BATCH LLM BIAS ANALYSIS EXPERIMENTS"
echo "========================================================================"
echo ""

# Create output and log directories
mkdir -p outputs/batch_experiments
mkdir -p logs

# Timestamp for this batch
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="outputs/batch_experiments/batch_$TIMESTAMP"
mkdir -p "$BATCH_DIR"

echo "Batch output directory: $BATCH_DIR"
echo ""

# Experiment 1: Popular prompt style
echo "========================================================================"
echo "Experiment 1: Popular prompt style"
echo "========================================================================"
DATASET_SIZE=1500 POOL_SIZE=75 K=15 PROMPT_STYLE=popular OUTPUT_DIR="$BATCH_DIR/exp1_popular" \
    ./run_experiment.sh 2>&1 | tee "$BATCH_DIR/exp1_popular.log"

# Experiment 2: Engaging prompt style
echo ""
echo "========================================================================"
echo "Experiment 2: Engaging prompt style"
echo "========================================================================"
DATASET_SIZE=1500 POOL_SIZE=75 K=15 PROMPT_STYLE=engaging OUTPUT_DIR="$BATCH_DIR/exp2_engaging" \
    ./run_experiment.sh 2>&1 | tee "$BATCH_DIR/exp2_engaging.log"

# Experiment 3: Informative prompt style
echo ""
echo "========================================================================"
echo "Experiment 3: Informative prompt style"
echo "========================================================================"
DATASET_SIZE=1500 POOL_SIZE=75 K=15 PROMPT_STYLE=informative OUTPUT_DIR="$BATCH_DIR/exp3_informative" \
    ./run_experiment.sh 2>&1 | tee "$BATCH_DIR/exp3_informative.log"

# Experiment 4: Controversial prompt style
echo ""
echo "========================================================================"
echo "Experiment 4: Controversial prompt style"
echo "========================================================================"
DATASET_SIZE=1500 POOL_SIZE=75 K=15 PROMPT_STYLE=controversial OUTPUT_DIR="$BATCH_DIR/exp4_controversial" \
    ./run_experiment.sh 2>&1 | tee "$BATCH_DIR/exp4_controversial.log"

# Summary
echo ""
echo "========================================================================"
echo "✅ BATCH EXPERIMENTS COMPLETE"
echo "========================================================================"
echo "Results saved to: $BATCH_DIR"
echo ""
echo "To compare results across experiments, run:"
echo "  python compare_experiments.py $BATCH_DIR/*/bias_analysis.json"
echo ""
