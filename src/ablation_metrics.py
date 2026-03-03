# src/ablation_metrics.py
#
# Usage:
#   python src/ablation_metrics.py
#   python src/ablation_metrics.py --input results/ablation_raw_mock.json

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict, Counter

parser = argparse.ArgumentParser()
parser.add_argument("--input", default=None, help="Path to raw results JSON (default: results/ablation_raw.json)")
args = parser.parse_args()

BASE         = Path(__file__).resolve().parent.parent
RESULTS_PATH = Path(args.input) if args.input else BASE / "results" / "ablation_raw.json"
SCORES_PATH  = BASE / "results" / "ablation_scores.json"

STRATEGY_ORDER = ["zero_shot", "few_shot", "chain_of_thought", "cot_self_consistency"]
STRATEGY_LABELS = {
    "zero_shot":            "Zero-Shot",
    "few_shot":             "Few-Shot",
    "chain_of_thought":     "Chain-of-Thought",
    "cot_self_consistency": "CoT + Self-Consistency",
}


# ── Answer extraction ─────────────────────────────────────────────────────────

def strip_markdown(text: str) -> str:
    """Remove markdown bold/italic so **$160** becomes $160."""
    text = re.sub(r"\*{1,3}", "", text)
    text = re.sub(r"_{1,3}", "", text)
    text = re.sub(r"`+",     "", text)
    return text


def extract_number(text: str) -> str | None:
    """
    Extract the final numeric answer from a model response.

    Priority order (highest to lowest confidence):
      1. Explicit answer keyword  — "Answer: 160", "The answer is $160"
      2. Bold number in original  — Gemini loves **160** or **$160**
      3. Number at end of text    — last thing said is usually the answer
      4. Last number anywhere     — lowest-confidence fallback
    """
    clean = strip_markdown(text)

    # 1. Explicit answer keyword patterns — highest confidence
    high_confidence = [
        r"[Aa]nswer[:\s]+\$?([\d,]+\.?\d*)",           # "Answer: 160"
        r"[Tt]he answer is[:\s]+\$?([\d,]+\.?\d*)",    # "The answer is 160"
        r"[Tt]otal[:\s]+\$?([\d,]+\.?\d*)",            # "Total: 160"
        r"[Tt]herefore[,\s]+\$?([\d,]+\.?\d*)",        # "Therefore, 160"
        r"=\s*\$?([\d,]+\.?\d*)\s*$",                  # "= 160" at end of line
    ]
    for pat in high_confidence:
        match = re.search(pat, clean)
        if match:
            return match.group(1).replace(",", "").strip()

    # 2. Bold number in original text (before stripping) — Gemini's **160**
    bold_numbers = re.findall(r"\*\*\$?([\d,]+\.?\d*)\*\*", text)
    if bold_numbers:
        return bold_numbers[-1].replace(",", "").strip()

    # 3. Number at end of text (most likely the answer in short responses)
    match = re.search(r"\$?([\d,]+\.?\d*)\s*[.\n]?\s*$", clean.strip())
    if match:
        val = match.group(1).replace(",", "").strip()
        if val and val != ".":
            return val

    # 4. Last number anywhere — absolute fallback
    numbers = re.findall(r"\$?([\d,]+\.?\d*)", clean)
    numbers = [n for n in numbers if n and n != "."]
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def majority_vote(responses: list) -> str | None:
    """For CoT+SC: extract a number from each response and return the most common."""
    extracted = [extract_number(r) for r in responses]
    extracted = [n for n in extracted if n is not None]
    if not extracted:
        return None
    return Counter(extracted).most_common(1)[0][0]


def normalize(s: str) -> str:
    """Normalize numbers for comparison: '160.00' == '160' == '160.0'"""
    s = s.strip().lower().replace(",", "")
    try:
        return str(float(s)).rstrip("0").rstrip(".")
    except ValueError:
        return s


def answers_match(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False
    return normalize(predicted) == normalize(ground_truth)


# ── Score computation ─────────────────────────────────────────────────────────

def compute_scores(results: dict) -> dict:
    counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for key, entry in results.items():
        strategy = entry["strategy"]
        model    = entry["model"]
        gt       = entry["ground_truth"]

        if strategy == "cot_self_consistency":
            predicted = majority_vote(entry["responses"])
        else:
            predicted = extract_number(entry["response"])

        is_correct = answers_match(predicted, gt)
        counts[strategy][model]["total"]   += 1
        counts[strategy][model]["correct"] += int(is_correct)

    scores = {}
    for strategy, models in counts.items():
        scores[strategy] = {}
        for model, c in models.items():
            acc = c["correct"] / c["total"] if c["total"] > 0 else 0
            scores[strategy][model] = {
                "correct":  c["correct"],
                "total":    c["total"],
                "accuracy": round(acc * 100, 1),
            }
    return scores


# ── Debug: show mismatches to verify extraction ───────────────────────────────

def show_extraction_samples(results: dict, n: int = 5):
    """Print first N wrong extractions so you can spot-check the logic."""
    print(f"\n🔍 Sample mismatches (first {n}) — verify extraction is working:")
    shown = 0
    for key, entry in results.items():
        if shown >= n:
            break
        strategy = entry["strategy"]
        gt = entry["ground_truth"]
        if strategy == "cot_self_consistency":
            predicted = majority_vote(entry["responses"])
            raw = entry["responses"][0][:150] if entry["responses"] else ""
        else:
            predicted = extract_number(entry["response"])
            raw = entry["response"][:150]
        if not answers_match(predicted, gt):
            print(f"  GT={gt:<8} predicted={str(predicted):<10} model={entry['model']} strategy={strategy}")
            print(f"  response: {raw!r}")
            print()
            shown += 1
    if shown == 0:
        print("  ✅ None found — extraction looks correct!")


# ── Pretty-print table ────────────────────────────────────────────────────────

def print_table(scores: dict, models: list):
    col_w = 26
    header = f"{'Strategy':<{col_w}}" + "".join(f"{m:>26}" for m in models)
    sep = "=" * len(header)
    print(f"\n{sep}")
    print("GSM8K Ablation — Accuracy (%)")
    print(sep)
    print(header)
    print("-" * len(header))
    for strategy in STRATEGY_ORDER:
        if strategy not in scores:
            continue
        label = STRATEGY_LABELS[strategy]
        row = f"{label:<{col_w}}"
        for m in models:
            if m in scores[strategy]:
                c    = scores[strategy][m]
                cell = f"{c['accuracy']:.1f}% ({c['correct']}/{c['total']})"
                row += f"{cell:>26}"
            else:
                row += f"{'N/A':>26}"
        print(row)
    print(sep)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not RESULTS_PATH.exists():
        print(f"❌ No results file found at {RESULTS_PATH}")
        print("   Run ablation_evaluate.py first.")
        return

    with open(RESULTS_PATH) as f:
        results = json.load(f)

    print(f"📂 Loaded {len(results)} entries from {RESULTS_PATH.name}")

    scores     = compute_scores(results)
    all_models = sorted({entry["model"] for entry in results.values()})

    print_table(scores, all_models)

    with open(SCORES_PATH, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\n✅ Scores saved to {SCORES_PATH}")

    print("\n📊 Best model per strategy:")
    for strategy in STRATEGY_ORDER:
        if strategy not in scores:
            continue
        best_model = max(scores[strategy], key=lambda m: scores[strategy][m]["accuracy"])
        best_acc   = scores[strategy][best_model]["accuracy"]
        print(f"   {STRATEGY_LABELS[strategy]:<30} → {best_model} ({best_acc}%)")

    show_extraction_samples(results)


if __name__ == "__main__":
    main()