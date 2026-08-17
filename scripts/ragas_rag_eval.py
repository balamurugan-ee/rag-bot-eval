"""
Ragas evaluation for RAG receptionist, testing the REAL /chat endpoint.

RETRIEVAL METRICS (evaluate quality of retrieved context):
  - Context Precision: Of the retrieved chunks, how many were relevant to 
    producing the expected answer?
  - Context Recall: Did retrieval fetch everything needed to support the 
    expected answer?

GENERATION METRICS (evaluate quality of generated answer):
  - Faithfulness: Does the answer only contain claims traceable to the
    retrieved context?
  - Answer Correctness: Does the answer match the expected answer, both
    factually (TP/FP/FN claim overlap) and semantically?

Answer Relevancy was dropped: its "noncommittal" flag zeroed out correct terse
answers (e.g. scored 0.00 despite 0.90 underlying similarity), and it penalized
answers for including extra correct detail beyond the literal question. Both
behaviors produced misleading scores rather than real quality signal.

The answer being graded comes from a real HTTP call to /chat, testing what's
actually deployed. Retrieved contexts come from an independent vector store
call since /chat only returns {response}.

This script captures full reasoning for each metric, not just final scores.

Output: Console output with reasoning, JSON dump (tests/ragas_results.json),
and interactive HTML report (tests/ragas_report.html).
"""
import asyncio
import csv
import json
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import requests
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerCorrectness,
    ContextPrecisionWithReference,
    ContextRecall
)
from ragas.metrics.collections.context_precision.metric import ContextPrecisionInput, ContextPrecisionOutput
from ragas.metrics.collections.context_recall.metric import ContextRecallInput, ContextRecallOutput

from src.vectordb.manager import VectorStoreManager
from src.config import settings

# File paths
CSV_PATH = PROJECT_ROOT / "tests" / "rag_ground_truth.csv"
JSON_OUTPUT_PATH = PROJECT_ROOT / "tests" / "ragas_results.json"
HTML_TEMPLATE_PATH = PROJECT_ROOT / "scripts" / "ragas_report_template.html"
HTML_OUTPUT_PATH = PROJECT_ROOT / "tests" / "ragas_report.html"
CHAT_URL = "http://localhost:8000/chat"

# CI gate: Answer Correctness is the one metric that compares directly against
# the gold answer regardless of category (on-topic, refusal, injection, tone),
# so it's used as the pass/fail threshold. Faithfulness/Relevancy/Precision/Recall
# are reported but not gated on -- they have real structural blind spots for
# refusal-type answers (a correct refusal isn't "grounded in retrieved context"
# by definition), so a low score there doesn't reliably mean a real regression.
CORRECTNESS_THRESHOLD = 0.7


def write_html_report(results):
    template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(results))
    HTML_OUTPUT_PATH.write_text(html, encoding="utf-8")


# Note: the combined GHA step summary (classification + RAG together) is
# written by scripts/write_pipeline_summary.py as the final step of the
# job, reading this script's JSON output -- not written here directly, so
# there's a single consolidated summary instead of two separate fragments.


def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return [
            {
                "question": row["question"].strip(),
                "expected_answer": row["expected_answer"].strip(),
                "category": row.get("category", "").strip(),
            }
            for row in csv.DictReader(f)
        ]


def get_real_answer(question: str) -> str:
    response = requests.post(CHAT_URL, json={"message": question}, timeout=30)
    response.raise_for_status()
    return response.json()["response"]


async def score_retrieval_metrics(metrics: dict, question: str, expected_answer: str, contexts: list[str]) -> dict:
    """
    Score retrieval metrics: Context Precision and Context Recall.
    These metrics evaluate the quality of retrieved context chunks.
    """
    results = {}
    
    # Context Precision - judges each chunk against expected answer
    metric = metrics['context_precision']
    verdicts = []
    precision_reasoning = []
    for context in contexts:
        input_data = ContextPrecisionInput(question=question, context=context, answer=expected_answer)
        result = await metric.llm.agenerate(metric.prompt.to_string(input_data), ContextPrecisionOutput)
        verdicts.append(result.verdict)
        precision_reasoning.append({
            "context_preview": context[:120] + ("..." if len(context) > 120 else ""),
            "relevant": bool(result.verdict),
            "reason": result.reason,
        })
    results['context_precision'] = {
        'score': float(metric._calculate_average_precision(verdicts)),
        'reasoning': precision_reasoning
    }

    # Context Recall - checks if context supports expected answer
    metric = metrics['context_recall']
    context_str = "\n".join(contexts)
    input_data = ContextRecallInput(question=question, context=context_str, answer=expected_answer)
    result = await metric.llm.agenerate(metric.prompt.to_string(input_data), ContextRecallOutput)
    if result.classifications:
        attributions = [c.attributed for c in result.classifications]
        results['context_recall'] = {
            'score': float(sum(attributions) / len(attributions)),
            'reasoning': [
                {"statement": c.statement, "attributed": bool(c.attributed), "reason": c.reason}
                for c in result.classifications
            ]
        }
    else:
        results['context_recall'] = {'score': float("nan"), 'reasoning': []}
    
    return results


async def score_generation_metrics(metrics: dict, question: str, answer: str, expected_answer: str, contexts: list[str]) -> dict:
    """
    Score generation metrics: Faithfulness and Answer Correctness.
    These metrics evaluate the quality of the generated answer.
    """
    results = {}
    context_str = "\n".join(contexts)

    # Faithfulness - does answer contain only claims traceable to context?
    metric = metrics['faithfulness']
    statements = await metric._create_statements(question, answer)
    if statements:
        verdicts = await metric._create_verdicts(statements, context_str)
        results['faithfulness'] = {
            'score': float(metric._compute_score(verdicts)),
            'reasoning': [
                {"statement": s.statement, "faithful": bool(s.verdict), "reason": s.reason}
                for s in verdicts.statements
            ]
        }
    else:
        results['faithfulness'] = {'score': float("nan"), 'reasoning': []}

    # Answer Correctness - does answer match expected answer factually and semantically?
    metric = metrics['answer_correctness']
    response_statements = await metric._generate_statements(question, answer)
    reference_statements = await metric._generate_statements(question, expected_answer)
    
    if response_statements and reference_statements:
        classification = await metric._classify_statements(question, response_statements, reference_statements)
        factuality_score = metric._compute_f1_score(classification)
        claim_reasoning = {
            "true_positive": [{"statement": s.statement, "reason": s.reason} for s in classification.TP],
            "false_positive": [{"statement": s.statement, "reason": s.reason} for s in classification.FP],
            "false_negative": [{"statement": s.statement, "reason": s.reason} for s in classification.FN],
        }
    else:
        factuality_score = 1.0
        claim_reasoning = {"true_positive": [], "false_positive": [], "false_negative": []}
    
    similarity_score = await metric._calculate_similarity(answer, expected_answer)
    results['answer_correctness'] = {
        'score': float(np.average([factuality_score, similarity_score], weights=metric.weights)),
        'reasoning': {
            "factuality_score": float(factuality_score),
            "similarity_score": float(similarity_score),
            **claim_reasoning,
        }
    }
    
    return results


def print_results(retrieval_scores: dict, generation_scores: dict):
    """Print organized results by metric category."""
    
    # Retrieval Metrics
    print("\n=== RETRIEVAL METRICS ===")
    
    print(f"\nContext Precision: {retrieval_scores['context_precision']['score']:.2f}")
    for r in retrieval_scores['context_precision']['reasoning']:
        print(f"  [{'RELEVANT' if r['relevant'] else 'NOT RELEVANT'}] {r['context_preview']}")
    
    print(f"\nContext Recall: {retrieval_scores['context_recall']['score']:.2f}")
    for r in retrieval_scores['context_recall']['reasoning']:
        print(f"  [{'ATTRIBUTED' if r['attributed'] else 'NOT ATTRIBUTED'}] {r['statement']}")
    
    # Generation Metrics
    print("\n=== GENERATION METRICS ===")
    
    print(f"\nFaithfulness: {generation_scores['faithfulness']['score']:.2f}")
    for r in generation_scores['faithfulness']['reasoning']:
        print(f"  [{'OK' if r['faithful'] else 'UNFAITHFUL'}] {r['statement']}")

    correctness = generation_scores['answer_correctness']
    print(f"\nAnswer Correctness: {correctness['score']:.2f}")
    print(f"  (factuality={correctness['reasoning']['factuality_score']:.2f}, similarity={correctness['reasoning']['similarity_score']:.2f})")
    for s in correctness['reasoning']["true_positive"]:
        print(f"  [MATCH] {s['statement']}")
    for s in correctness['reasoning']["false_positive"]:
        print(f"  [EXTRA/WRONG] {s['statement']}")
    for s in correctness['reasoning']["false_negative"]:
        print(f"  [MISSING] {s['statement']}")


async def main():
    # Initialize services
    vector_store = VectorStoreManager()
    vector_store.initialize()

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    ragas_llm = llm_factory("gpt-4o-mini", client=client)
    ragas_embeddings = embedding_factory("openai", "text-embedding-3-small", client=client)

    # Initialize metrics categorized by type
    retrieval_metrics = {
        'context_precision': ContextPrecisionWithReference(llm=ragas_llm),
        'context_recall': ContextRecall(llm=ragas_llm),
    }
    
    generation_metrics = {
        'faithfulness': Faithfulness(llm=ragas_llm),
        'answer_correctness': AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings),
    }

    # Process each test case
    results = []
    for row in load_rows():
        question = row["question"]
        answer = get_real_answer(question)
        contexts = [c.page_content for c in vector_store.similarity_search(question, k=3)]
        expected_answer = row["expected_answer"]

        # Score metrics by category
        retrieval_scores = await score_retrieval_metrics(retrieval_metrics, question, expected_answer, contexts)
        generation_scores = await score_generation_metrics(generation_metrics, question, answer, expected_answer, contexts)

        # Display results
        print(f"\nQ: {question}")
        print(f"Expected: {expected_answer}")
        print(f"A: {answer}")
        print_results(retrieval_scores, generation_scores)
        print("\n" + "=" * 80 + "\n")

        # Store results
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "category": row["category"],
            "answer": answer,
            "retrieved_contexts": contexts,
            "retrieval_metrics": retrieval_scores,
            "generation_metrics": generation_scores,
            # Flatten for backward compatibility
            **retrieval_scores,
            **generation_scores,
        })

    # Save outputs
    JSON_OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Full results with reasoning written to {JSON_OUTPUT_PATH}")

    write_html_report(results)
    print(f"Interactive HTML report written to {HTML_OUTPUT_PATH}")

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'Question':<40} | {'Prec':>6} {'Recall':>6} | {'Faith':>6} {'Correct':>6}")
    print("-" * 100)
    for r in results:
        q = r['question'][:38] + ".." if len(r['question']) > 40 else r['question']
        print(f"{q:<40} | {r['context_precision']['score']:>6.2f} {r['context_recall']['score']:>6.2f} | "
              f"{r['faithfulness']['score']:>6.2f} {r['answer_correctness']['score']:>6.2f}")
    print("=" * 100)

    # Per-category breakdown (informational -- helps explain a threshold failure at a glance)
    categories = sorted({r["category"] for r in results if r["category"]})
    if categories:
        print("\nPer-category Answer Correctness:")
        for cat in categories:
            cat_rows = [r for r in results if r["category"] == cat]
            cat_avg = sum(r["answer_correctness"]["score"] for r in cat_rows) / len(cat_rows)
            print(f"  {cat:<20} {cat_avg:.2f}  ({len(cat_rows)} questions)")

    # CI gate: fail the run if average Answer Correctness drops below threshold.
    # See CORRECTNESS_THRESHOLD comment above for why this metric specifically.
    avg_correctness = sum(r["answer_correctness"]["score"] for r in results) / len(results)
    print(f"\nAverage Answer Correctness: {avg_correctness:.3f} (threshold: {CORRECTNESS_THRESHOLD})")

    if avg_correctness < CORRECTNESS_THRESHOLD:
        print(f"FAILED: average Answer Correctness {avg_correctness:.3f} is below threshold {CORRECTNESS_THRESHOLD}")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    asyncio.run(main())
