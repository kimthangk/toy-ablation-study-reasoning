# src/sanity_test.py
#
# Verifies all API keys work before running the full experiment.
#
# Usage:
#   python src/sanity_test.py

import os
import sys
from pathlib import Path
import openai
from dotenv import load_dotenv

load_dotenv()

models = {
    "gemini-2.5-flash-lite": {
        "model_id": "gemini-2.5-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key":      os.getenv("GEMINI_API_KEY"),
        "kwargs":   {"reasoning_effort": "none"},
    },
    "gemini-2.5-flash-lite:thinking": {
        "model_id": "gemini-2.5-flash-lite",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key":      os.getenv("GEMINI_API_KEY"),
        "kwargs":   {},
    },
    "llama-3.3-70b-versatile": {
        "model_id": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "key":      os.getenv("GROQ_API_KEY"),
        "kwargs":   {},
    },
}

print("\n🔑 Sanity check — testing all API keys...\n")

all_passed = True
for model, cfg in models.items():
    try:
        client = openai.OpenAI(api_key=cfg["key"], base_url=cfg["base_url"])
        r = client.chat.completions.create(
            model=cfg["model_id"],
            messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            max_tokens=10,
            temperature=0,
            **cfg["kwargs"],
        )
        content = r.choices[0].message.content
        print(f"  ✅ {model}: {repr(content)}")
    except Exception as e:
        print(f"  ❌ {model}: {e}")
        all_passed = False

print()
if all_passed:
    print("✅ All keys working — ready to run the experiment!")
    print("   python src/ablation_evaluate.py --dry-run")
    print("   python src/ablation_evaluate.py")
else:
    print("❌ Fix the errors above before running the experiment.")
