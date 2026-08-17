"""
Consolidated GitHub Actions step summary for the eval pipeline.

Reads both stages' JSON output (promptfoo classification results + Ragas RAG
results) and writes ONE combined Markdown summary: an overall table with
both stages' pass/fail and scores, a callout for any individual failures,
and per-stage detail tables collapsed behind <details> so the page stays
scannable regardless of how many test cases exist.

This is the single source of truth for the pipeline's step summary -- the
individual eval scripts (check_promptfoo_threshold.py, ragas_rag_eval.py)
only print to the console and set their own exit codes; they don't write
to $GITHUB_STEP_SUMMARY themselves, so there's one coherent summary instead
of two disconnected fragments.

Runs as the final step of the eval job with `if: always()`, so it still
produces a useful summary (noting what did/didn't run) even if an earlier
stage failed and later stages were skipped.
"""
import json
import os
from pathlib import Path

CLASSIFICATION_RESULTS_PATH = Path("promptfoo-classification-results.json")
RAG_RESULTS_PATH = Path("tests") / "ragas_results.json"

CLASSIFICATION_THRESHOLD = 0.90
RAG_THRESHOLD = 0.70

CLASSIFICATION_ICON = "\U0001F3F7️"  # 🏷️
RAG_ICON = "\U0001F4DA"  # 📚
PASS_ICON = "✅"  # ✅
FAIL_ICON = "❌"  # ❌
WARN_ICON = "⚠️"  # ⚠️


def load_classification():
    if not CLASSIFICATION_RESULTS_PATH.exists():
        return None
    data = json.loads(CLASSIFICATION_RESULTS_PATH.read_text(encoding="utf-8"))
    rows = data["results"]["results"]
    total = len(rows)
    passed = sum(1 for r in rows if r["gradingResult"]["pass"])
    accuracy = passed / total if total else 0.0
    return {"rows": rows, "total": total, "passed": passed, "accuracy": accuracy}


def load_rag():
    if not RAG_RESULTS_PATH.exists():
        return None
    results = json.loads(RAG_RESULTS_PATH.read_text(encoding="utf-8"))
    avg_correctness = sum(r["answer_correctness"]["score"] for r in results) / len(results) if results else 0.0
    return {"results": results, "avg_correctness": avg_correctness}


def status_badge(is_pass):
    return f"{PASS_ICON} PASS" if is_pass else f"{FAIL_ICON} FAIL"


def main():
    classification = load_classification()
    rag = load_rag()

    lines = ["# Eval Pipeline Summary", ""]

    # --- Overall table: both stages, at a glance ---
    lines += ["| Stage | Status | Score | Threshold |", "| --- | --- | --- | --- |"]

    if classification:
        c_pass = classification["accuracy"] >= CLASSIFICATION_THRESHOLD
        lines.append(
            f"| {CLASSIFICATION_ICON} Classification | {status_badge(c_pass)} | "
            f"{classification['accuracy']:.1%} ({classification['passed']}/{classification['total']}) | "
            f"{CLASSIFICATION_THRESHOLD:.0%} |"
        )
    else:
        c_pass = False
        lines.append(f"| {CLASSIFICATION_ICON} Classification | {WARN_ICON} DID NOT RUN | -- | -- |")

    if rag:
        r_pass = rag["avg_correctness"] >= RAG_THRESHOLD
        lines.append(
            f"| {RAG_ICON} RAG | {status_badge(r_pass)} | {rag['avg_correctness']:.1%} | {RAG_THRESHOLD:.0%} |"
        )
    else:
        r_pass = False
        lines.append(f"| {RAG_ICON} RAG | {WARN_ICON} DID NOT RUN (skipped if the classification gate failed) | -- | -- |")

    overall_pass = c_pass and r_pass
    lines += ["", f"**Overall: {status_badge(overall_pass)}**", ""]

    # --- Failures callout: only the rows that actually failed, no scanning required ---
    classification_failures = [r for r in classification["rows"] if not r["gradingResult"]["pass"]] if classification else []
    rag_failures = [r for r in rag["results"] if r["answer_correctness"]["score"] < RAG_THRESHOLD] if rag else []

    if classification_failures or rag_failures:
        lines += [f"## {WARN_ICON} Failures", ""]
        for r in classification_failures:
            message = r["prompt"]["raw"][:70]
            expected = r["gradingResult"]["componentResults"][0]["assertion"].get("value", "")
            got = r["response"]["output"]
            lines.append(f"- {CLASSIFICATION_ICON} \"{message}\" -- expected `{expected}`, got `{got}`")
        for r in rag_failures:
            q = r["question"][:70]
            lines.append(f"- {RAG_ICON} \"{q}\" -- Answer Correctness {r['answer_correctness']['score']:.2f}")
        lines.append("")

    # --- Classification detail (collapsed) ---
    if classification:
        lines += [f"<details><summary>{CLASSIFICATION_ICON} Classification Detail ({classification['total']} questions)</summary>", ""]
        lines += ["| Message | Expected | Got | Result |", "| --- | --- | --- | --- |"]
        for r in classification["rows"]:
            message = r["prompt"]["raw"].replace("|", "\\|")[:60]
            expected = str(r["gradingResult"]["componentResults"][0]["assertion"].get("value", "")).replace("|", "\\|")
            got = str(r["response"]["output"]).replace("|", "\\|")[:80]
            status = status_badge(r["gradingResult"]["pass"])
            lines.append(f"| {message} | {expected} | {got} | {status} |")
        lines += ["", "</details>", ""]

    # --- RAG detail (collapsed) ---
    if rag:
        lines += [f"<details><summary>{RAG_ICON} RAG Detail ({len(rag['results'])} questions)</summary>", ""]
        lines += ["| Question | Precision | Recall | Faithfulness | Correctness |", "| --- | --- | --- | --- | --- |"]
        for r in rag["results"]:
            q = r["question"].replace("|", "\\|")[:60]
            lines.append(
                f"| {q} | {r['context_precision']['score']:.2f} | {r['context_recall']['score']:.2f} | "
                f"{r['faithfulness']['score']:.2f} | {r['answer_correctness']['score']:.2f} |"
            )
        lines.append("")

        categories = sorted({r["category"] for r in rag["results"] if r.get("category")})
        if categories:
            lines += ["### Per-category Answer Correctness", "", "| Category | Avg Correctness | Questions |", "| --- | --- | --- |"]
            for cat in categories:
                cat_rows = [r for r in rag["results"] if r["category"] == cat]
                cat_avg = sum(r["answer_correctness"]["score"] for r in cat_rows) / len(cat_rows)
                lines.append(f"| {cat} | {cat_avg:.2f} | {len(cat_rows)} |")
        lines += ["", "</details>"]

    summary = "\n".join(lines)
    print(summary)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary + "\n")


if __name__ == "__main__":
    main()
