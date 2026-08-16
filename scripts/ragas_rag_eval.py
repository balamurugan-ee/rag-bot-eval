"""
Ragas eval for the RAG receptionist, testing the REAL /chat endpoint.

Three reference-free metrics, all scoring the system's own answer/retrieval
against each other -- no gold/expected answer needed:
  - Faithfulness: does the answer only contain claims traceable to context?
  - Answer Relevancy: does the answer actually address the question asked?
  - Context Precision: of the retrieved chunks, how many were relevant?

Questions come from tests/rag_ground_truth.csv -- Excel-editable, same
pattern as tests/classification_ground_truth.csv.

The answer being graded always comes from a real HTTP call to /chat -- not
from calling ReceptionistBot directly -- so this tests what's actually
deployed. retrieved_contexts for Faithfulness and Context Precision come
from an independent, read-only vector store call, since /chat only returns
{response} by design. Answer Relevancy needs no context at all.

Each metric's .score()/.ascore() shortcut only returns the final number and
throws away its own intermediate reasoning -- so this script calls the same
internal pieces ascore() calls (statement generation + NLI verdicts for
faithfulness, per-chunk verdicts for context precision, reverse-engineered
questions + similarity for answer relevancy) directly, to capture *why* each
score came out the way it did, not just the number.

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
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecisionWithoutReference
from ragas.metrics.collections.context_precision.metric import ContextPrecisionInput, ContextPrecisionOutput
from ragas.metrics.collections.answer_relevancy.metric import AnswerRelevanceInput, AnswerRelevanceOutput

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


def load_questions():
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        return [row["question"].strip() for row in csv.DictReader(f)]


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


async def score_context_precision(metric: ContextPrecisionWithoutReference, question: str, answer: str, contexts: list[str]):
    verdicts = []
    reasoning = []
    for context in contexts:
        input_data = ContextPrecisionInput(question=question, context=context, answer=answer)
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


async def main():
    vector_store = VectorStoreManager()
    vector_store.initialize()

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    ragas_llm = llm_factory("gpt-4o-mini", client=client)
    ragas_embeddings = embedding_factory("openai", "text-embedding-3-small", client=client)

    faithfulness = Faithfulness(llm=ragas_llm)
    answer_relevancy = AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    context_precision = ContextPrecisionWithoutReference(llm=ragas_llm)

    results = []
    for question in load_questions():
        answer = get_real_answer(question)
        contexts = [c.page_content for c in vector_store.similarity_search(question, k=3)]

        faith_score, faith_reasoning = await score_faithfulness(faithfulness, question, answer, contexts)
        relevancy_score, relevancy_reasoning = await score_answer_relevancy(answer_relevancy, question, answer)
        precision_score, precision_reasoning = await score_context_precision(context_precision, question, answer, contexts)

        print(f"Q: {question}")
        print(f"A: {answer}")
        print(f"\nFaithfulness: {faith_score:.2f}")
        for r in faith_reasoning:
            print(f"  [{'OK' if r['faithful'] else 'UNFAITHFUL'}] {r['statement']}")
            print(f"      why: {r['reason']}")
        print(f"\nAnswer Relevancy: {relevancy_score:.2f}")
        for r in relevancy_reasoning:
            flag = " (NONCOMMITTAL)" if r["noncommittal"] else ""
            print(f"  sim={r['similarity_to_original']:.2f}{flag}  reverse-engineered Q: {r['generated_question']}")
        print(f"\nContext Precision: {precision_score:.2f}")
        for r in precision_reasoning:
            print(f"  [{'RELEVANT' if r['relevant'] else 'NOT RELEVANT'}] {r['context_preview']}")
            print(f"      why: {r['reason']}")
        print("\n" + "=" * 80 + "\n")

        results.append({
            "question": question,
            "answer": answer,
            "retrieved_contexts": contexts,
            "faithfulness": {"score": faith_score, "reasoning": faith_reasoning},
            "answer_relevancy": {"score": relevancy_score, "reasoning": relevancy_reasoning},
            "context_precision": {"score": precision_score, "reasoning": precision_reasoning},
        })

    JSON_OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Full results with reasoning written to {JSON_OUTPUT_PATH}")

    write_html_report(results)
    print(f"Interactive HTML report written to {HTML_OUTPUT_PATH}")

    print("\n--- Summary ---")
    print(f"{'Question':<55} {'Faith':>7} {'Relev':>7} {'Prec':>7}")
    for r in results:
        print(f"{r['question'][:53]:<55} {r['faithfulness']['score']:>7.2f} {r['answer_relevancy']['score']:>7.2f} {r['context_precision']['score']:>7.2f}")


if __name__ == "__main__":
    asyncio.run(main())
