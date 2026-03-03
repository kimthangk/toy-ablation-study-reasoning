# Prompting Strategy Ablation Study on GSM8K

An ablation study comparing 4 prompting strategies across 3 LLM configurations on 50 GSM8K math word problems.

## Overview

This project evaluates how prompting strategy affects LLM accuracy on grade-school math reasoning. It runs all combinations of models × strategies in parallel with automatic checkpointing, then scores results and prints a comparison table.

**Models tested:**
| Alias | Provider | Notes |
|---|---|---|
| `gemini-2.5-flash-lite` | Google AI | Reasoning / thinking OFF |
| `gemini-2.5-flash-lite:thinking` | Google AI | Reasoning / thinking ON (default) |
| `llama-3.3-70b-versatile` | Groq | — |

**Strategies tested:**
| Strategy | Description |
|---|---|
| Zero-Shot | Question only; model answers directly |
| Few-Shot | 3 worked examples prepended before the question |
| Chain-of-Thought (CoT) | 3 step-by-step examples; model shows reasoning |
| CoT + Self-Consistency | CoT run 5× per question; majority vote selects final answer |

**Dataset:** [GSM8K](https://huggingface.co/datasets/gsm8k) — 50 questions from the test split.

---

## Project Structure

```
project2/
├── data/
│   └── prompts.py           # 4 prompting strategies + STRATEGIES registry
├── src/
│   ├── sanity_test.py       # Validates all API keys before a real run
│   ├── ablation_evaluate.py # Main script — calls APIs, saves raw results
│   └── ablation_metrics.py  # Scores accuracy, prints table, saves JSON
├── results/
│   ├── ablation_raw.json        # Raw model responses (real run)
│   ├── ablation_raw_mock.json   # Raw model responses (dry run)
│   └── ablation_scores.json     # Accuracy scores per strategy/model
├── .env                     # API keys (never commit)
├── .gitignore
└── requirements.txt
```

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Add API keys to `.env`**
```
GEMINI_API_KEY=your_google_ai_key_here
GROQ_API_KEY=your_groq_key_here
```

Free API keys (no credit card required):
- Gemini: https://aistudio.google.com/apikey
- Groq: https://console.groq.com/keys

---

## Usage

### Step 1 — Validate API keys
```bash
python src/sanity_test.py
```
Sends a single `2+2` prompt to each model and confirms all keys work.

### Step 2 — Run the evaluation

**Dry run** (no API calls, instant, uses mock responses):
```bash
python src/ablation_evaluate.py --dry-run
```

**Real run** (calls all 3 APIs, ~750 total API calls):
```bash
python src/ablation_evaluate.py
```

Results are saved to `results/ablation_raw.json` after every call. If the script is interrupted, re-running it resumes from where it left off — already-completed entries are skipped.

### Step 3 — Compute accuracy metrics
```bash
# From real run results
python src/ablation_metrics.py

# From dry run / mock results
python src/ablation_metrics.py --input results/ablation_raw_mock.json
```

Prints an accuracy table and saves `results/ablation_scores.json`.

---

## API Call Volume

| Component | Count |
|---|---|
| Questions | 50 |
| Strategies | 4 |
| Models | 3 |
| Base calls | 600 |
| CoT+SC multiplier (5×) | +750 |
| **Total calls** | **~1,350** |

Rate limits respected: Gemini 10 RPM, Groq 25 RPM. Models run in parallel threads.

---

## Architecture

**Data flow:**

```
data/prompts.py
    └─ STRATEGIES dict (4 prompt-builder functions)
           │
           ▼
src/ablation_evaluate.py
    └─ Loads GSM8K via HuggingFace datasets
    └─ ThreadPoolExecutor (1 thread per model)
    └─ Checkpoints to results/ablation_raw.json after every call
           │
           ▼
src/ablation_metrics.py
    └─ Extracts numeric answers via regex (priority: explicit keyword → bold → end-of-text → fallback)
    └─ Majority vote for CoT+SC
    └─ Writes results/ablation_scores.json
```

**Key design choices:**
- `temperature=0` and `max_tokens=500` are identical across all models and strategies for fair comparison.
- Both Gemini aliases share one API client; `reasoning_effort="none"` is passed only for the non-thinking variant.
- Checkpointing keys are `{q_idx}|{strategy}|{model}` — safe to stop and resume mid-run.
- Errored entries are cleared and re-queued automatically on resume.

---

## Example Results

```
==============================================================================================
GSM8K Ablation — Accuracy (%)
==============================================================================================
Strategy                  gemini-2.5-flash-lite  gemini-2.5-flash-lite:thinking  llama-3.3-70b-versatile
----------------------------------------------------------------------------------------------
Zero-Shot                      36.0% (18/50)              48.0% (24/50)              56.0% (28/50)
Few-Shot                       54.0% (27/50)              62.0% (31/50)              52.0% (26/50)
Chain-of-Thought               56.0% (28/50)              70.0% (35/50)              52.0% (26/50)
CoT + Self-Consistency         52.0% (26/50)              72.0% (36/50)              56.0% (28/50)
==============================================================================================
```

*(Numbers above are illustrative — run the experiment to generate your own results.)*
