# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

```bash
pip install -r requirements.txt
# Add API keys to .env (see .env for required keys: GOOGLE_API_KEY, GROQ_API_KEY)
```

## Commands

**Validate API keys before running:**
```bash
python src/sanity_test.py
```

**Run the ablation evaluation:**
```bash
# Dry run — no API calls, uses mock data
python src/ablation_evaluate.py --dry-run

# Real run — calls Gemini and Llama APIs
python src/ablation_evaluate.py
```

**Compute accuracy metrics from results:**
```bash
python src/ablation_metrics.py

# Score mock results instead
python src/ablation_metrics.py --input results/ablation_raw_mock.json
```

## Architecture

This is an LLM ablation study comparing 4 prompting strategies × 2 models on 50 GSM8K math word problems.

**Data flow:**
1. `data/prompts.py` — defines the 4 prompting strategies (Zero-Shot, Few-Shot, CoT, CoT+Self-Consistency) and a `STRATEGY_REGISTRY` dict mapping names to prompt-builder functions. `COT_SC_RUNS = 5` controls self-consistency sample count.
2. `src/ablation_evaluate.py` — main script. Loads GSM8K via HuggingFace `datasets`, then runs both models **in parallel** using `ThreadPoolExecutor` (one thread per model). Writes checkpoint JSON after each API call so runs can be resumed. Output: `results/ablation_raw.json`.
3. `src/ablation_metrics.py` — reads raw JSON, extracts numeric answers via regex, applies majority voting for CoT+SC, computes accuracy per strategy/model, and writes `results/ablation_scores.json`.

**Models and API clients:**
- **Gemini 2.5 Flash Lite (reasoning OFF)** — alias `gemini-2.5-flash-lite`. Google AI OpenAI-compatible endpoint. Passes `reasoning_effort="none"` to disable thinking. Rate limit: 10 RPM.
- **Gemini 2.5 Flash Lite (reasoning ON)** — alias `gemini-2.5-flash-lite:thinking`. Same client and model ID as above; no `reasoning_effort` override so thinking is enabled by default. Rate limit: 10 RPM.
- **Llama 3.3 70B** — accessed via Groq OpenAI-compatible endpoint (`https://api.groq.com/openai/v1`). Rate limit: 25 RPM.

All three use `temperature=0` for deterministic outputs. Both Gemini aliases share one API client (`GEMINI_API_KEY`); the `MODEL_IDS` dict maps both to `"gemini-2.5-flash-lite"`.

**Total API calls:** 50 questions × 4 strategies × 3 models = 600 calls (CoT+SC adds 5× multiplier for its strategy, totaling ~750 additional calls for that condition).

**Checkpointing:** Results are saved to JSON after each call. If the script is interrupted, re-running it skips already-completed entries (keyed by `model + strategy + question_index`).
