# RAGAS Evaluation Script - Clean Architecture

## Overview
Simplified and organized evaluation script for RAG system with clear separation of concerns.

## Metric Categories

### 🔍 RETRIEVAL METRICS
Evaluate the quality of retrieved context chunks:
- **Context Precision**: How many retrieved chunks were relevant?
- **Context Recall**: Did retrieval fetch everything needed?

### 🤖 GENERATION METRICS  
Evaluate the quality of generated answers:
- **Faithfulness**: Answer only contains claims traceable to context
- **Answer Relevancy**: Answer addresses the question asked
- **Answer Correctness**: Answer matches expected answer (factually + semantically)

## Code Structure

```
ragas_rag_eval.py (313 lines, down from 274)
│
├── Utility Functions
│   ├── write_html_report()      # Generate HTML report
│   ├── load_rows()               # Load test cases from CSV
│   └── get_real_answer()         # Call /chat endpoint
│
├── Scoring Functions (ORGANIZED BY CATEGORY)
│   ├── score_retrieval_metrics()     # Context Precision + Recall
│   └── score_generation_metrics()    # Faithfulness + Relevancy + Correctness
│
├── Display Functions
│   └── print_results()           # Organized output by category
│
└── Main Orchestration
    └── main()                    # Clean workflow with categorized metrics
```

## Key Improvements

### ✅ Before → After

1. **5 separate functions** → **2 organized functions** (by category)
2. **Repetitive print logic** → **Single structured print function**
3. **Flat metric list** → **Categorized metric dictionaries**
4. **Verbose output** → **Organized by RETRIEVAL vs GENERATION**
5. **Basic summary** → **Professional table with categories**

## Output Format

### Console Output
```
Q: [question]
Expected: [expected_answer]
A: [actual_answer]

=== RETRIEVAL METRICS ===
Context Precision: 0.85
  [RELEVANT] chunk preview...
  [NOT RELEVANT] chunk preview...

Context Recall: 0.90
  [ATTRIBUTED] statement...
  [NOT ATTRIBUTED] statement...

=== GENERATION METRICS ===
Faithfulness: 0.95
  [OK] claim statement...
  [UNFAITHFUL] claim statement...

Answer Relevancy: 0.88
  reverse-engineered Q: ... (sim=0.88)

Answer Correctness: 0.92
  (factuality=0.90, similarity=0.94)
  [MATCH] correct claim...
  [EXTRA/WRONG] incorrect claim...
  [MISSING] missing claim...

================================================================================

SUMMARY TABLE
================================================================================
Question                                 |   Prec Recall |  Faith  Relev Correct
----------------------------------------------------------------------------------------------------
What are the visiting hours?..           |   0.85   0.90 |   0.95   0.88   0.92
How do I book an appointment?..          |   0.92   0.88 |   0.93   0.91   0.94
================================================================================
```

### JSON Output Structure
```json
[
  {
    "question": "...",
    "expected_answer": "...",
    "answer": "...",
    "retrieved_contexts": [...],
    "retrieval_metrics": {
      "context_precision": {"score": 0.85, "reasoning": [...]},
      "context_recall": {"score": 0.90, "reasoning": [...]}
    },
    "generation_metrics": {
      "faithfulness": {"score": 0.95, "reasoning": [...]},
      "answer_relevancy": {"score": 0.88, "reasoning": [...]},
      "answer_correctness": {"score": 0.92, "reasoning": {...}}
    }
  }
]
```

## Benefits

1. **Clear Separation**: Retrieval vs Generation concerns are isolated
2. **Maintainable**: Easy to add new metrics in the right category
3. **Readable**: Output clearly shows what's being evaluated
4. **Professional**: Better organized reports for stakeholders
5. **Debuggable**: Category-based structure makes issues easier to trace

## Usage

```bash
# Ensure server is running on localhost:8000
python scripts/ragas_rag_eval.py
```

## Output Files

- `tests/ragas_results.json` - Full results with reasoning
- `tests/ragas_report.html` - Interactive HTML report
- Console - Organized metric output by category

