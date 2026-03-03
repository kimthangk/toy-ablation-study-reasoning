# src/ablation_evaluate.py
#
# What it does:
#   - Loads 50 GSM8K test questions
#   - Runs 3 models IN PARALLEL (separate threads, separate API keys, separate rate limits)
#     · gemini-2.5-flash-lite          (reasoning OFF)
#     · gemini-2.5-flash-lite:thinking (reasoning ON)
#     · llama-3.3-70b-versatile
#   - 4 strategies per model: zero-shot, few-shot, CoT, CoT+SC (5 runs, majority vote)
#   - Checkpoints after every call — safe to stop and resume
#
# Free API keys needed (all no credit card):
#   - Gemini:  https://aistudio.google.com/apikey
#   - Groq:    https://console.groq.com/keys
#
# Usage:
#   python src/ablation_evaluate.py              ← real run
#   python src/ablation_evaluate.py --dry-run    ← mock run, no API calls, instant

import argparse
import json
import os
import re
import sys
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm
import openai

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.prompts import STRATEGIES, COT_SC_RUNS

# ── Config ────────────────────────────────────────────────────────────────────
NUM_QUESTIONS = 50
TEMPERATURE   = 0      # Always 0 — reproducibility
MAX_TOKENS    = 500    # Same for ALL strategies — fairness

RESULTS_PATH      = Path(__file__).resolve().parent.parent / "results" / "ablation_raw.json"
MOCK_RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "ablation_raw_mock.json"
RESULTS_PATH.parent.mkdir(exist_ok=True)

# ── One client per model, each with its own key + base URL ───────────────────
# Both Gemini variants share one client; the alias key differs from the API model ID.
_gemini_client = openai.OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODELS = {
    "gemini-2.5-flash-lite":          _gemini_client,
    "gemini-2.5-flash-lite:thinking": _gemini_client,
    "llama-3.3-70b-versatile":        openai.OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    ),
}

# Actual API model IDs (both Gemini aliases map to the same model string)
MODEL_IDS = {
    "gemini-2.5-flash-lite":          "gemini-2.5-flash-lite",
    "gemini-2.5-flash-lite:thinking": "gemini-2.5-flash-lite",
    "llama-3.3-70b-versatile":        "llama-3.3-70b-versatile",
}

# Rate limits (requests per minute, conservative)
RATE_LIMITS = {
    "gemini-2.5-flash-lite":          10,
    "gemini-2.5-flash-lite:thinking": 10,
    "llama-3.3-70b-versatile":        25,
}

# Extra kwargs per model
MODEL_KWARGS = {
    "gemini-2.5-flash-lite":          {"reasoning_effort": "none"},   # thinking OFF
    "gemini-2.5-flash-lite:thinking": {},                              # thinking ON (default)
    "llama-3.3-70b-versatile":        {},
}

# ── Mock responses (used in --dry-run mode) ───────────────────────────────────
MOCK_RESPONSES = {
    "zero_shot":            "The answer is 42.",
    "few_shot":             "42",
    "chain_of_thought":     "Step 1: ... Step 2: ... Answer: 42",
    "cot_self_consistency": "Step 1: ... Step 2: ... Answer: 42",
}

# ── Thread-safe checkpoint ────────────────────────────────────────────────────
_results_lock = threading.Lock()

def load_results(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}

def save_results(results: dict, path: Path):
    with _results_lock:
        with open(path, "w") as f:
            json.dump(results, f, indent=2)

def make_key(q_idx: int, strategy: str, model: str) -> str:
    safe = model.replace("/", "__").replace(":", "--")
    return f"{q_idx}|{strategy}|{safe}"

# ── API call ──────────────────────────────────────────────────────────────────
def call_model(model_name: str, prompt: str, dry_run: bool, strategy_name: str) -> str:
    if dry_run:
        return MOCK_RESPONSES[strategy_name]
    client   = MODELS[model_name]
    model_id = MODEL_IDS[model_name]
    extra    = MODEL_KWARGS[model_name]
    response = client.chat.completions.create(
        model=model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        **extra,
    )
    return response.choices[0].message.content.strip()

# ── GSM8K helpers ─────────────────────────────────────────────────────────────
def extract_gsm8k_answer(text: str) -> str:
    match = re.search(r"####\s*([\d,\.]+)", text)
    return match.group(1).replace(",", "").strip() if match else text.strip()

def load_gsm8k(n: int = NUM_QUESTIONS):
    from datasets import load_dataset
    print(f"📥 Loading {n} GSM8K questions...")
    ds = load_dataset("gsm8k", "main", split=f"test[:{n}]")
    return [{"question": item["question"], "ground_truth": extract_gsm8k_answer(item["answer"])} for item in ds]

# ── Per-model worker ──────────────────────────────────────────────────────────
def run_model_worker(model_name: str, questions: list, results: dict,
                     pbar: tqdm, dry_run: bool, save_path: Path):
    sleep_between = 0 if dry_run else 60 / RATE_LIMITS[model_name]

    for q_idx, q in enumerate(questions):
        for strategy_name, prompt_fn in STRATEGIES.items():
            key = make_key(q_idx, strategy_name, model_name)

            with _results_lock:
                already_done = key in results
            if already_done:
                pbar.update(1)
                continue

            prompt = prompt_fn(q["question"])

            if strategy_name == "cot_self_consistency":
                responses = []
                for _ in range(COT_SC_RUNS):
                    try:
                        responses.append(call_model(model_name, prompt, dry_run, strategy_name))
                    except Exception as e:
                        responses.append(f"ERROR: {e}")
                    if not dry_run:
                        time.sleep(sleep_between)
                entry = {
                    "q_idx": q_idx, "question": q["question"],
                    "ground_truth": q["ground_truth"],
                    "strategy": strategy_name, "model": model_name,
                    "responses": responses,
                }
            else:
                try:
                    response = call_model(model_name, prompt, dry_run, strategy_name)
                except Exception as e:
                    response = f"ERROR: {e}"
                entry = {
                    "q_idx": q_idx, "question": q["question"],
                    "ground_truth": q["ground_truth"],
                    "strategy": strategy_name, "model": model_name,
                    "response": response,
                }

            with _results_lock:
                results[key] = entry
            save_results(results, save_path)
            if not dry_run:
                time.sleep(sleep_between)
            pbar.update(1)

# ── Main ──────────────────────────────────────────────────────────────────────
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Mock all API calls — no keys needed, runs instantly")
    args = parser.parse_args()

    dry_run    = args.dry_run
    save_path  = MOCK_RESULTS_PATH if dry_run else RESULTS_PATH
    questions  = load_gsm8k()
    results    = load_results(save_path)

# Clear errored entries so they get re-run on resume
    error_keys = [
        k for k, v in results.items()
        if "ERROR" in str(v.get("response", ""))
        or any("ERROR" in r for r in v.get("responses", []))
    ]
    if error_keys:
        print(f"🧹 Clearing {len(error_keys)} errored entries for re-run...")
        for k in error_keys:
            del results[k]
        save_results(results, save_path)

    total        = len(questions) * len(STRATEGIES) * len(MODELS)
    already_done = len(results)

    mode_label = "🧪 DRY RUN (mock responses, no API calls)" if dry_run else "🚀 LIVE RUN"
    print(f"\n{mode_label}")
    print(f"📊 Total calls : {total}  ({len(MODELS)} models × {len(STRATEGIES)} strategies × {len(questions)} questions)")
    print(f"✅ Cached      : {already_done}")
    print(f"⏳ Remaining   : {total - already_done}")
    if dry_run:
        print(f"💾 Saving to   : {MOCK_RESULTS_PATH.name}\n")
    else:
        print(f"💾 Saving to   : {RESULTS_PATH.name}\n")

    with tqdm(total=total, initial=already_done, desc="Overall progress") as pbar:
        with ThreadPoolExecutor(max_workers=len(MODELS)) as executor:
            futures = {
                executor.submit(
                    run_model_worker, model_name, questions,
                    results, pbar, dry_run, save_path
                ): model_name
                for model_name in MODELS
            }
            for future in as_completed(futures):
                model_name = futures[future]
                try:
                    future.result()
                    print(f"\n✅ {model_name} — done")
                except Exception as e:
                    print(f"\n❌ {model_name} — error: {e}")

    print(f"\n🎉 All done! Results saved to {save_path}")
    print(f"   Total entries: {len(results)}")
    if dry_run:
        print("\n✅ Dry run passed — pipeline works end-to-end")
        print("   When ready for real run: python src/ablation_evaluate.py")
    else:
        print("\nNext step: python src/ablation_metrics.py")

if __name__ == "__main__":
    run()