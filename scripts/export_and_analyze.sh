#!/bin/bash
# Export Promptfoo results and generate confusion matrix

set -e

CONFIG=${1:-promptfoo.classify.yaml}
OUTPUT_DIR="outputs"
RESULTS_JSON="$OUTPUT_DIR/promptfoo_results.json"

mkdir -p "$OUTPUT_DIR"

echo "Running Promptfoo evaluation..."
promptfoo eval -c "$CONFIG" --output "$RESULTS_JSON" --no-cache

echo ""
echo "Generating confusion matrix..."
python scripts/build_confusion_matrix.py --input "$RESULTS_JSON"


echo ""
echo "✅ Complete! Check outputs/ directory"


