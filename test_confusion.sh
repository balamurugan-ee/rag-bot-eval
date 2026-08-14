#!/bin/bash
# Test the confusion matrix generation

cd /Users/balamurugan/Documents/ee-projects/work/demo

echo "Testing confusion matrix generation..."
python scripts/build_confusion_matrix.py --input outputs/promptfoo_results.json

echo ""
echo "Done! Check the outputs above."

