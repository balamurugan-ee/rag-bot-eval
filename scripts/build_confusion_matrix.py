#!/usr/bin/env python3
"""
scripts/build_confusion_matrix.py

Reads Promptfoo results (JSON or CSV), extracts expected and predicted labels,
and generates:
 - outputs/confusion_matrix.csv
 - outputs/confusion_matrix.png
 - outputs/classification_report.txt
 - outputs/metrics_summary.json

Usage:
  python scripts/build_confusion_matrix.py --input promptfoo_results.json
  python scripts/build_confusion_matrix.py --input promptfoo_results.csv
"""
import argparse
import json
from pathlib import Path
from collections import deque
import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support
)
import seaborn as sns
import matplotlib.pyplot as plt

# Common column/field names for expected and predicted labels
COMMON_EXPECTED_COLS = [
    "__expected", "expected", "label", "ground_truth", "gold", 
    "expected_label", "__label__", "y_true", "assert"
]
COMMON_PRED_COLS = [
    "predicted", "prediction", "output", "result", "response", 
    "predicted_label", "provider_output", "y_pred", "department"
]

def normalize_label(l):
    """Normalize label to consistent format"""
    if pd.isna(l) or l is None:
        return "UNKNOWN"
    v = str(l).strip()
    # Remove quotes if present
    v = v.strip('"').strip("'")
    return v if v else "UNKNOWN"

def find_in_dict_recursive(obj, candidate_keys, max_depth=10):
    """Breadth-first search for any candidate key in nested dict/list"""
    candidate_keys = {k.lower() for k in candidate_keys}
    q = deque([(obj, 0)])
    
    while q:
        cur, depth = q.popleft()
        if depth > max_depth:
            continue
            
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k.lower() in candidate_keys:
                    return v
                q.append((v, depth + 1))
        elif isinstance(cur, list):
            for item in cur:
                q.append((item, depth + 1))
    return None

def parse_csv(path: Path, expected_col=None, pred_col=None):
    """Parse CSV file and extract expected/predicted labels"""
    df = pd.read_csv(path, dtype=str)
    
    # Auto-detect expected column
    if expected_col is None:
        for c in COMMON_EXPECTED_COLS:
            if c in df.columns:
                expected_col = c
                break
        if expected_col is None and len(df.columns) >= 2:
            expected_col = df.columns[1]
    
    # Auto-detect predicted column
    if pred_col is None:
        for c in COMMON_PRED_COLS:
            if c in df.columns:
                pred_col = c
                break
    
    if expected_col not in df.columns:
        raise RuntimeError(
            f"Could not find expected column. Available: {list(df.columns)}. "
            f"Use --expected-col to specify."
        )
    
    if pred_col not in df.columns:
        raise RuntimeError(
            f"Could not find predicted column. Available: {list(df.columns)}. "
            f"Use --pred-col to specify."
        )
    
    y_true = df[expected_col].astype(str).apply(normalize_label).tolist()
    y_pred = df[pred_col].astype(str).apply(normalize_label).tolist()
    messages = df["message"].astype(str).tolist() if "message" in df.columns else [""] * len(y_true)
    
    print(f"✓ Parsed CSV: {len(y_true)} records")
    print(f"  Expected column: {expected_col}")
    print(f"  Predicted column: {pred_col}")
    
    return messages, y_true, y_pred

def parse_json(path: Path, expected_field=None, pred_field=None):
    """Parse JSON file and extract expected/predicted labels"""
    with open(path, "r") as fh:
        data = json.load(fh)
    
    # Normalize to list of test items
    if isinstance(data, dict):
        # Promptfoo format: data.results.results is the actual results array
        if "results" in data and isinstance(data["results"], dict):
            if "results" in data["results"] and isinstance(data["results"]["results"], list):
                items = data["results"]["results"]
            else:
                items = [data]
        # Try other common keys for lists
        elif "results" in data and isinstance(data["results"], list):
            items = data["results"]
        elif "tests" in data and isinstance(data["tests"], list):
            items = data["tests"]
        elif "items" in data and isinstance(data["items"], list):
            items = data["items"]
        else:
            items = [data]
    else:
        items = data if isinstance(data, list) else [data]
    
    messages = []
    y_true = []
    y_pred = []
    
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        
        # Extract message
        msg = ""
        vars_obj = item.get("vars", {})
        if isinstance(vars_obj, dict):
            msg = vars_obj.get("message", "")
        
        # Extract expected label
        exp_val = None
        if expected_field:
            # Follow dot path
            parts = expected_field.split(".")
            val = item
            for part in parts:
                if isinstance(val, dict):
                    val = val.get(part)
                elif isinstance(val, list) and part.isdigit():
                    val = val[int(part)]
                else:
                    val = None
                    break
            if val is not None:
                exp_val = normalize_label(val)
        
        # Auto-detect expected
        if exp_val is None:
            # Promptfoo specific: testCase.assert[0].value
            test_case = item.get("testCase", {})
            if isinstance(test_case, dict):
                assert_list = test_case.get("assert", [])
                if isinstance(assert_list, list) and len(assert_list) > 0:
                    assert_val = assert_list[0].get("value")
                    if assert_val:
                        exp_val = normalize_label(assert_val)
        
        # Check vars if still not found
        if exp_val is None:
            if isinstance(vars_obj, dict):
                for key in COMMON_EXPECTED_COLS:
                    if key in vars_obj:
                        exp_val = normalize_label(vars_obj[key])
                        break
        
        # Deep search for expected
        if exp_val is None:
            val = find_in_dict_recursive(item, COMMON_EXPECTED_COLS)
            if val is not None:
                exp_val = normalize_label(val)
        
        # Extract predicted label
        pred_val = None
        if pred_field:
            # Follow dot path
            parts = pred_field.split(".")
            val = item
            for part in parts:
                if isinstance(val, dict):
                    val = val.get(part)
                elif isinstance(val, list) and part.isdigit():
                    val = val[int(part)]
                else:
                    val = None
                    break
            if val is not None:
                pred_val = normalize_label(val)
        
        # Auto-detect predicted
        if pred_val is None:
            # Promptfoo specific: response.output
            response_obj = item.get("response", {})
            if isinstance(response_obj, dict) and "output" in response_obj:
                pred_val = normalize_label(response_obj["output"])
        
        # Check other common response paths
        if pred_val is None:
            for path in [
                ["response", "department"],
                ["output", "department"],
                ["result", "department"],
                ["provider", "output"],
                ["output"],
                ["response"],
            ]:
                val = item
                for key in path:
                    if isinstance(val, dict) and key in val:
                        val = val[key]
                    else:
                        val = None
                        break
                if val is not None:
                    if isinstance(val, dict) and "department" in val:
                        pred_val = normalize_label(val["department"])
                        break
                    elif isinstance(val, str):
                        pred_val = normalize_label(val)
                        break
        
        # Deep search for predicted
        if pred_val is None:
            val = find_in_dict_recursive(item, COMMON_PRED_COLS)
            if val is not None:
                pred_val = normalize_label(val)
        
        messages.append(msg)
        y_true.append(exp_val if exp_val else "UNKNOWN")
        y_pred.append(pred_val if pred_val else "UNKNOWN")
    
    print(f"✓ Parsed JSON: {len(y_true)} records")
    return messages, y_true, y_pred

def compute_metrics(y_true, y_pred, labels):
    """Compute all classification metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='weighted', zero_division=0
    )
    
    return {
        "accuracy": float(accuracy),
        "macro_avg": {
            "precision": float(macro_precision),
            "recall": float(macro_recall),
            "f1_score": float(macro_f1)
        },
        "weighted_avg": {
            "precision": float(weighted_precision),
            "recall": float(weighted_recall),
            "f1_score": float(weighted_f1)
        },
        "per_class": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i])
            }
            for i in range(len(labels))
        }
    }

def save_outputs(y_true, y_pred, messages, out_dir: Path):
    """Generate and save all outputs"""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    labels = sorted(list(set(y_true) | set(y_pred)))
    
    # 1. Confusion Matrix CSV
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(
        cm, 
        index=[f"true:{l}" for l in labels], 
        columns=[f"pred:{l}" for l in labels]
    )
    cm_csv = out_dir / "confusion_matrix.csv"
    cm_df.to_csv(cm_csv)
    print(f"\n✓ Saved confusion matrix CSV: {cm_csv}")
    
    # 2. Classification Report
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    report_path = out_dir / "classification_report.txt"
    report_path.write_text(report)
    print(f"✓ Saved classification report: {report_path}")
    
    # 3. Metrics Summary JSON
    metrics = compute_metrics(y_true, y_pred, labels)
    metrics_path = out_dir / "metrics_summary.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved metrics summary: {metrics_path}")
    
    # 4. Detailed Results CSV (with errors highlighted)
    results_df = pd.DataFrame({
        "message": messages,
        "expected": y_true,
        "predicted": y_pred,
        "correct": [t == p for t, p in zip(y_true, y_pred)]
    })
    results_csv = out_dir / "detailed_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"✓ Saved detailed results: {results_csv}")
    
    # 5. Confusion Matrix Heatmap
    plt.figure(figsize=(max(8, len(labels) * 0.8), max(6, len(labels) * 0.6)))
    sns.set(font_scale=1.0)
    ax = sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight='bold')
    ax.set_ylabel("True Label", fontsize=12, fontweight='bold')
    ax.set_title("Confusion Matrix", fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    img_path = out_dir / "confusion_matrix.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved confusion matrix heatmap: {img_path}")
    
    # 6. Write to GitHub Step Summary if in CI
    write_github_summary(metrics, cm_df, results_df, out_dir)
    
    # Print summary to console
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS SUMMARY")
    print("=" * 70)
    print(f"\nAccuracy: {metrics['accuracy']:.3f}")
    print(f"\nMacro Average:")
    print(f"  Precision: {metrics['macro_avg']['precision']:.3f}")
    print(f"  Recall:    {metrics['macro_avg']['recall']:.3f}")
    print(f"  F1-Score:  {metrics['macro_avg']['f1_score']:.3f}")
    print(f"\nWeighted Average:")
    print(f"  Precision: {metrics['weighted_avg']['precision']:.3f}")
    print(f"  Recall:    {metrics['weighted_avg']['recall']:.3f}")
    print(f"  F1-Score:  {metrics['weighted_avg']['f1_score']:.3f}")
    print("\n" + "=" * 70)
    print("\nPer-Class Metrics:")
    print("-" * 70)
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")
    print("-" * 70)
    for label in labels:
        m = metrics['per_class'][label]
        print(f"{label:<20} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1_score']:>10.3f} {m['support']:>10}")
    print("=" * 70)
    
    # Print errors
    errors = results_df[~results_df["correct"]]
    if len(errors) > 0:
        print(f"\n⚠️  Found {len(errors)} misclassifications:")
        print("-" * 70)
        for idx, row in errors.iterrows():
            print(f"\nMessage: {row['message']}")
            print(f"  Expected:  {row['expected']}")
            print(f"  Predicted: {row['predicted']}")
        print("-" * 70)


def write_github_summary(metrics, cm_df, results_df, out_dir: Path):
    """Write confusion matrix summary to GitHub Actions step summary"""
    import os
    
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    
    img_path = out_dir / "confusion_matrix.png"
    
    lines = ["", "## 📊 Confusion Matrix", ""]
    lines.append(f"**Accuracy: {metrics['accuracy']:.2%}** | Precision: {metrics['macro_avg']['precision']:.2%} | Recall: {metrics['macro_avg']['recall']:.2%} | F1: {metrics['macro_avg']['f1_score']:.2%}")
    lines.append("")
    lines.append(f"![Confusion Matrix]({img_path})")
    lines.append("")
    
    errors = results_df[results_df["correct"] == False]
    if len(errors) > 0:
        lines.append(f"⚠️ **{len(errors)} misclassifications** - See artifacts for details")
        lines.append("")
    
    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"✓ Written to GitHub Step Summary")
    except Exception as e:
        print(f"⚠️  Could not write to GitHub Step Summary: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Build confusion matrix from Promptfoo results"
    )
    parser.add_argument(
        "--input", "-i", 
        required=True,
        help="Path to Promptfoo results (JSON or CSV)"
    )
    parser.add_argument(
        "--expected-col",
        help="CSV: column name for expected labels"
    )
    parser.add_argument(
        "--pred-col",
        help="CSV: column name for predicted labels"
    )
    parser.add_argument(
        "--expected-field",
        help="JSON: dot-path to expected field (e.g., vars.__expected)"
    )
    parser.add_argument(
        "--pred-field",
        help="JSON: dot-path to predicted field (e.g., output.department)"
    )
    parser.add_argument(
        "--out-dir",
        default="outputs",
        help="Output directory (default: outputs)"
    )
    
    args = parser.parse_args()
    
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"❌ Input file not found: {inp}")
    
    print(f"Reading Promptfoo results from: {inp}")
    
    # Parse input file
    ext = inp.suffix.lower()
    if ext in {".csv", ".tsv"}:
        messages, y_true, y_pred = parse_csv(
            inp, 
            expected_col=args.expected_col,
            pred_col=args.pred_col
        )
    else:
        messages, y_true, y_pred = parse_json(
            inp,
            expected_field=args.expected_field,
            pred_field=args.pred_field
        )
    
    # Save all outputs
    out_dir = Path(args.out_dir)
    save_outputs(y_true, y_pred, messages, out_dir)
    
    print(f"\n✅ All outputs saved to: {out_dir}/")

if __name__ == "__main__":
    main()

