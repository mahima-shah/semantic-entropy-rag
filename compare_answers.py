"""
compare_answers.py

Compares the semantic meaning of two generated answers using an LLM.

This module is part of the semantic entropy-inspired confidence layer.
Instead of comparing wording, it asks the language model whether two
answers communicate the same underlying meaning.
"""

from llm import ask_llm


# -----------------------------
# Semantic Comparison
# -----------------------------

def compare_meaning(answer_a: str, answer_b: str) -> str:
    """
    Compare whether two answers express the same meaning.

    The LLM is instructed to return only YES or NO. If an unexpected
    response is received, the result is marked as UNCLEAR.

    Args:
        answer_a:
            First generated answer.

        answer_b:
            Second generated answer.

    Returns:
        "YES", "NO", or "UNCLEAR".
    """

    prompt = f"""
You are comparing two answers.

Answer A:
{answer_a}

Answer B:
{answer_b}

Do these answers communicate the same meaning?

Reply with exactly one word:
YES
or
NO
"""

    result = ask_llm(prompt, max_tokens=3)
    result = result.strip().upper()

    if "YES" in result:
        return "YES"

    if "NO" in result:
        return "NO"

    return "UNCLEAR"


# -----------------------------
# Command-Line Example
# -----------------------------

if __name__ == "__main__":

    answer_1 = """
Force majeure refers to events beyond a party's reasonable control,
such as fire, flood, earthquake, war, terrorism, riots,
government actions, and similar events.
"""

    answer_2 = """
Force majeure means circumstances outside a party's control that may
excuse delay or non-performance, including natural disasters,
war, terrorism, and government action.
"""

    comparison = compare_meaning(answer_1, answer_2)

    print("\nSame meaning?")
    print(comparison)