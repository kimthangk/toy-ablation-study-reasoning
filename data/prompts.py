# data/prompts.py
# The 4 prompting strategies for the ablation study.
# Each function takes a question string and returns the full prompt to send to the LLM.

# --- Few-shot examples (shared across strategies that use them) ---
FEW_SHOT_EXAMPLES = [
    {
        "question": "A train travels 60 miles per hour for 2.5 hours. How far does it travel?",
        "answer": "150 miles"
    },
    {
        "question": "Maria has 48 apples. She gives 1/4 to her friend and eats 3 herself. How many does she have left?",
        "answer": "33 apples"
    },
    {
        "question": "A rectangle has a width of 7 cm and a length of 12 cm. What is its area?",
        "answer": "84 square centimeters"
    },
]

FEW_SHOT_COT_EXAMPLES = [
    {
        "question": "A train travels 60 miles per hour for 2.5 hours. How far does it travel?",
        "reasoning": "Speed = 60 mph, Time = 2.5 hours. Distance = Speed × Time = 60 × 2.5 = 150 miles.",
        "answer": "150 miles"
    },
    {
        "question": "Maria has 48 apples. She gives 1/4 to her friend and eats 3 herself. How many does she have left?",
        "reasoning": "1/4 of 48 = 12. She gives away 12. 48 - 12 = 36 remaining. She eats 3. 36 - 3 = 33.",
        "answer": "33 apples"
    },
    {
        "question": "A rectangle has a width of 7 cm and a length of 12 cm. What is its area?",
        "reasoning": "Area = width × length = 7 × 12 = 84.",
        "answer": "84 square centimeters"
    },
]


def zero_shot_prompt(question: str) -> str:
    """Strategy 1: Zero-shot — just the question, no examples or hints."""
    return (
        f"{question}\n\n"
        "Provide only the final numerical answer. Do not show your work."
    )


def few_shot_prompt(question: str) -> str:
    """Strategy 2: Few-shot — 3 worked examples before the question."""
    examples_text = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_text += f"Q: {ex['question']}\nA: {ex['answer']}\n\n"
    return (
        f"{examples_text}"
        f"Q: {question}\n"
        "A: Provide only the final numerical answer."
    )


def chain_of_thought_prompt(question: str) -> str:
    """Strategy 3: Chain-of-Thought — prompts the model to reason step by step."""
    examples_text = ""
    for ex in FEW_SHOT_COT_EXAMPLES:
        examples_text += (
            f"Q: {ex['question']}\n"
            f"Reasoning: {ex['reasoning']}\n"
            f"Answer: {ex['answer']}\n\n"
        )
    return (
        f"{examples_text}"
        f"Q: {question}\n"
        "Let's think step by step. Show your reasoning, then end with 'Answer: <number>'."
    )


def cot_self_consistency_prompt(question: str) -> str:
    """Strategy 4: CoT + Self-Consistency — same as CoT; caller runs this 5x and takes majority vote."""
    # Identical prompt to CoT — self-consistency is handled in ablation_evaluate.py
    # by running this prompt N times and majority-voting the answers.
    return chain_of_thought_prompt(question)


# Registry: maps strategy name → prompt function
STRATEGIES = {
    "zero_shot": zero_shot_prompt,
    "few_shot": few_shot_prompt,
    "chain_of_thought": chain_of_thought_prompt,
    "cot_self_consistency": cot_self_consistency_prompt,
}

COT_SC_RUNS = 5  # Number of times to run CoT+SC per question
