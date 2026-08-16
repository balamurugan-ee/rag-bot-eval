"""
Ragas eval for the RAG receptionist, testing the REAL /chat endpoint.

Five metrics -- the two reference-based ones need the expected_answer
column in tests/rag_ground_truth.csv:
  - Faithfulness (reference-free): does the answer only contain claims
    traceable to the retrieved context?
  - Answer Relevancy (reference-free): does the answer actually address
    the question asked?
  - Context Precision (reference-based): of the retrieved chunks, how many
    were relevant to producing the expected answer?
  - Context Recall (reference-based): did retrieval fetch everything needed
    to support the expected answer?
  - Answer Correctness (reference-based): does the answer match the
    expected answer, factually (TP/FP/FN claim overlap) and semantically?

The answer being graded always comes from a real HTTP call to /chat -- not
from calling ReceptionistBot directly -- so this tests what's actually
deployed. retrieved_contexts come from an independent, read-only vector
store call, since /chat only returns {response} by design.

Each metric's .score()/.ascore() shortcut only returns the final number and
throws away its own intermediate reasoning -- so this script calls the same
internal pieces ascore() calls directly, to capture *why* each score came
out the way it did, not just the number.

Output: printed per-question reasoning, a full JSON dump, and an interactive
HTML report (tests/ragas_report.html) -- regenerated on every run from
scripts/ragas_report_template.html, so the report is never a stale one-off.
"""
import asyncio
import csv
import json
from pathlib import Path

import numpy as np
import requests
from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    AnswerCorrectness,
)
from ragas.metrics.collections.context_precision.metric import ContextPrecisionInput, ContextPrecisionOutput
from ragas.metrics.collections.answer_relevancy.metric import AnswerRelevanceInput, AnswerRelevanceOutput
from ragas.metrics.collections.context_recall.metric import ContextRecallInput, ContextRecallOutput

from src.vectordb.manager import VectorStoreManager
from src.config import settings

CSV_PATH = Path(__file__).parent.parent / "tests" / "rag_ground_truth.csv"
JSON_OUTPUT_PATH = Path(__file__).parent.parent / "tests" / "ragas_results.json"
HTML_TEMPLATE_PATH = Path(__file__).parent / "ragas_report_template.html"
HTML_OUTPUT_PATH = Path(__file__).parent.parent / "tests" / "ragas_report.html"
CHAT_URL = "http://localhost:8000/chat"


def write_html_report(results):
    template = HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(results))
    HTML_OUTPUT_PATH.write_text(html, encoding="utf-8")


def load_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return [
            {"question": row["question"].strip(), "expected_answer": row["expected_answer"].strip()}
            for row in csv.DictReader(f)
        ]


def get_real_answer(question: str) -> str:
    response = requests.post(CHAT_URL, json={"message": question}, timeout=30)
    response.raise_for_status()
    return response.json()["response"]


async def score_faithfulness(metric: Faithfulness, question: str, answer: str, contexts: list[str]):
    statements = await metric._create_statements(question, answer)
    if not statements:
        return float("nan"), []

    context_str = "\n".join(contexts)
    verdicts = await metric._create_verdicts(statements, context_str)
    score = metric._compute_score(verdicts)

    reasoning = [
        {"statement": s.statement, "faithful": bool(s.verdict), "reason": s.reason}
        for s in verdicts.statements
    ]
    return float(score), reasoning


async def score_answer_relevancy(metric: AnswerRelevancy, question: str, answer: str):
    generated_questions = []
    noncommittal_flags = []

    for _ in range(metric.strictness):
        input_data = AnswerRelevanceInput(response=answer)
        prompt_string = metric.prompt.to_string(input_data)
        result = await metric.llm.agenerate(prompt_string, AnswerRelevanceOutput)
        if result.question:
            generated_questions.append(result.question)
            noncommittal_flags.append(result.noncommittal)

    if not generated_questions:
        return 0.0, []

    all_noncommittal = bool(np.all(noncommittal_flags))
    question_vec = np.asarray(await metric.embeddings.aembed_text(question)).reshape(1, -1)
    gen_question_vec = np.asarray(await metric.embeddings.aembed_texts(generated_questions)).reshape(len(generated_questions), -1)
    norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(question_vec, axis=1)
    cosine_sim = np.dot(gen_question_vec, question_vec.T).reshape(-1) / norm

    score = float(cosine_sim.mean() * int(not all_noncommittal))
    reasoning = [
        {"generated_question": q, "similarity_to_original": float(sim), "noncommittal": bool(nc)}
        for q, sim, nc in zip(generated_questions, cosine_sim, noncommittal_flags)
    ]
    return score, reasoning


async def score_context_precision(metric: ContextPrecisionWithReference, question: str, expected_answer: str, contexts: list[str]):
    """Judges each retrieved chunk against the EXPECTED answer, not the system's own answer."""
    verdicts = []
    reasoning = []
    for context in contexts:
        input_data = ContextPrecisionInput(question=question, context=context, answer=expected_answer)
        prompt_string = metric.prompt.to_string(input_data)
        result = await metric.llm.agenerate(prompt_string, ContextPrecisionOutput)
        verdicts.append(result.verdict)
        reasoning.append({
            "context_preview": context[:120] + ("..." if len(context) > 120 else ""),
            "relevant": bool(result.verdict),
            "reason": result.reason,
        })

    score = metric._calculate_average_precision(verdicts)
    return float(score), reasoning


async def score_context_recall(metric: ContextRecall, question: str, expected_answer: str, contexts: list[str]):
    """Checks whether the retrieved context supports every claim in the EXPECTED answer."""
    context_str = "\n".join(contexts)
    input_data = ContextRecallInput(question=question, context=context_str, answer=expected_answer)
    prompt_string = metric.prompt.to_string(input_data)
    result = await metric.llm.agenerate(prompt_string, ContextRecallOutput)

    if not result.classifications:
        return float("nan"), []

    attributions = [c.attributed for c in result.classifications]
    score = sum(attributions) / len(attributions)

    reasoning = [
        {"statement": c.statement, "attributed": bool(c.attributed), "reason": c.reason}
        for c in result.classifications
    ]
    return float(score), reasoning


async def score_answer_correctness(metric: AnswerCorrectness, question: str, answer: str, expected_answer: str):
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
    final_score = float(np.average([factuality_score, similarity_score], weights=metric.weights))

    reasoning = {
        "factuality_score": float(factuality_score),
        "similarity_score": float(similarity_score),
        **claim_reasoning,
    }
    return final_score, reasoning


async def main():
    vector_store = VectorStoreManager()
    vector_store.initialize()

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    ragas_llm = llm_factory("gpt-4o-mini", client=client)
    ragas_embeddings = embedding_factory("openai", "text-embedding-3-small", client=client)

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    context_precision = ContextPrecisionWithReference(llm=ragas_llm)
    context_recall = ContextRecall(llm=ragas_llm)
    answer_correctness = AnswerCorrectness(llm=ragas_llm, embeddings=ragas_embeddings)

    results = []
    for row in load_rows():
        question = row["question"]
        expected_answer = row["expected_answer"]

        answer = get_real_answer(question)
        contexts = [c.page_content for c in vector_store.similarity_search(question, k=3)]

        faith_score, faith_reasoning = await score_faithfulness(faithfulness, question, answer, contexts)
        relevancy_score, relevancy_reasoning = await score_answer_relevancy(answer_relevancy, question, answer)
        precision_score, precision_reasoning = await score_context_precision(context_precision, question, expected_answer, contexts)
        recall_score, recall_reasoning = await score_context_recall(context_recall, question, expected_answer, contexts)
        correctness_score, correctness_reasoning = await score_answer_correctness(answer_correctness, question, answer, expected_answer)

        print(f"Q: {question}")
        print(f"Expected: {expected_answer}")
        print(f"A: {answer}")
        print(f"\nFaithfulness: {faith_score:.2f}")
        for r in faith_reasoning:
            print(f"  [{'OK' if r['faithful'] else 'UNFAITHFUL'}] {r['statement']}")
        print(f"\nAnswer Relevancy: {relevancy_score:.2f}")
        for r in relevancy_reasoning[:1]:
            print(f"  reverse-engineered Q: {r['generated_question']} (sim={r['similarity_to_original']:.2f})")
        print(f"\nContext Precision: {precision_score:.2f}")
        for r in precision_reasoning:
            print(f"  [{'RELEVANT' if r['relevant'] else 'NOT RELEVANT'}] {r['context_preview']}")
        print(f"\nContext Recall: {recall_score:.2f}")
        for r in recall_reasoning:
            print(f"  [{'ATTRIBUTED' if r['attributed'] else 'NOT ATTRIBUTED'}] {r['statement']}")
        print(f"\nAnswer Correctness: {correctness_score:.2f} (factuality={correctness_reasoning['factuality_score']:.2f}, similarity={correctness_reasoning['similarity_score']:.2f})")
        for s in correctness_reasoning["true_positive"]:
            print(f"  [MATCH] {s['statement']}")
        for s in correctness_reasoning["false_positive"]:
            print(f"  [EXTRA/WRONG] {s['statement']}")
        for s in correctness_reasoning["false_negative"]:
            print(f"  [MISSING] {s['statement']}")
        print("\n" + "=" * 80 + "\n")

        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "answer": answer,
            "retrieved_contexts": contexts,
            "faithfulness": {"score": faith_score, "reasoning": faith_reasoning},
            "answer_relevancy": {"score": relevancy_score, "reasoning": relevancy_reasoning},
            "context_precision": {"score": precision_score, "reasoning": precision_reasoning},
            "context_recall": {"score": recall_score, "reasoning": recall_reasoning},
            "answer_correctness": {"score": correctness_score, "reasoning": correctness_reasoning},
        })

    JSON_OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Full results with reasoning written to {JSON_OUTPUT_PATH}")

    write_html_report(results)
    print(f"Interactive HTML report written to {HTML_OUTPUT_PATH}")

    print("\n--- Summary ---")
    print(f"{'Question':<45} {'Faith':>7} {'Relev':>7} {'Prec':>7} {'Recall':>7} {'Correct':>7}")
    for r in results:
        print(f"{r['question'][:43]:<45} {r['faithfulness']['score']:>7.2f} {r['answer_relevancy']['score']:>7.2f} {r['context_precision']['score']:>7.2f} {r['context_recall']['score']:>7.2f} {r['answer_correctness']['score']:>7.2f}")


if __name__ == "__main__":
    asyncio.run(main())
