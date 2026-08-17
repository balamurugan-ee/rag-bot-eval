"""
Quality gate for the promptfoo classification eval.

Reads promptfoo's own JSON output (produced via `promptfoo eval -o <path>`),
computes the pass rate explicitly, prints it, and exits non-zero if accuracy
drops below ACCURACY_THRESHOLD. That exit code is what actually stops the
pipeline -- the downstream RAG stage and deploy job both depend on this
step succeeding (see .github/workflows/eval-pipeline.yml).

The combined step summary (classification + RAG together) is written
separately by scripts/write_pipeline_summary.py at the end of the job, so
this script only prints to the console and sets the exit code.

Usage: python scripts/check_promptfoo_threshold.py <results.json>
"""
import json
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

    print(f"Classification accuracy: {accuracy:.1%} ({passed}/{total} passed)")
    for r in rows:
        message = r["prompt"]["raw"][:60]
        status = "PASS" if r["gradingResult"]["pass"] else "FAIL"
        print(f"  [{status}] {message}")

    print(f"\nAverage accuracy: {accuracy:.3f} (threshold: {ACCURACY_THRESHOLD})")
    if accuracy < ACCURACY_THRESHOLD:
        print(f"FAILED: accuracy {accuracy:.1%} is below threshold {ACCURACY_THRESHOLD:.0%}")
        sys.exit(1)
    print("PASSED")


if __name__ == "__main__":
    main()
