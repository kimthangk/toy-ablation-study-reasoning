project2/
├── data/
│   └── prompts.py          ← Strategy templates (zero-shot, few-shot, CoT, CoT+SC)
├── src/
│   ├── ablation_evaluate.py  ← Calls all 3 APIs, saves results
│   └── ablation_metrics.py   ← Scores accuracy, generates comparison table
├── results/                  ← Auto-created when you run
└── .env                      ← Your API keys (never commit this!)
├── .gitignore                ← Protects your keys from git
└── requirements.txt          ← Python dependencies