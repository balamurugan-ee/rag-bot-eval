"""
Quality gate for the promptfoo classification eval.

Reads promptfoo's own JSON output (produced via `promptfoo eval -o <path>`),
computes the pass rate explicitly, writes a Markdown summary to the GitHub
Actions step summary when running in CI, and exits non-zero if accuracy
drops below ACCURACY_THRESHOLD. That exit code is what actually stops the
pipeline -- the downstream RAG eval and deploy stages both depend on this
job succeeding (see .github/workflows/eval-pipeline.yml).

Usage: python scripts/check_promptfoo_threshold.py <results.json>
"""
import json
import os
import sys
from pathlib import Path

ACCURACY_THRESHOLD = 0.90


def main():
    if len(sys.argv) < 2:
        print("Usage: check_promptfoo_threshold.py <promptfoo-results.json>")
        sys.exit(2)

    results_path = Path(sys.argv[1])
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = data["results"]["results"]

    total = len(rows)
    passed = sum(1 for r in rows if r["gradingResult"]["pass"])
    accuracy = passed / total if total else 0.0

    lines = [
        "## Classification Eval Results",
        "",
        f"**Accuracy: {accuracy:.1%}** ({passed}/{total} passed) -- threshold: {ACCURACY_THRESHOLD:.0%}",
        "",
        "| Message | Expected | Got | Result |",
        "| --- | --- | --- | --- |",
    ]
    for r in rows:
        message = r["prompt"]["raw"].replace("|", "\\|")[:60]
        expected = str(r["gradingResult"]["componentResults"][0]["assertion"].get("value", "")).replace("|", "\\|")
        got = str(r["response"]["output"]).replace("|", "\\|")[:80]
        status = "PASS" if r["gradingResult"]["pass"] else "FAIL"
        lines.append(f"| {message} | {expected} | {got} | {status} |")

    summary = "\n".join(lines)
    print(summary)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write(summary + "\n")

    print(f"\nAverage accuracy: {accuracy:.3f} (threshold: {ACCURACY_THRESHOLD})")
    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAILED: accuracy {accuracy:.1%} is below threshold {ACCURACY_THRESHOLD:.0%}")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
