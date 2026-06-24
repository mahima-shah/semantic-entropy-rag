from llm import ask_llm


def compare_meaning(answer_a, answer_b):
    prompt = f"""
You are comparing two answers.

Answer A:
{answer_a}

Answer B:
{answer_b}

Do these answers communicate the same meaning?

Reply with exactly one word: YES or NO.
"""

    result = ask_llm(prompt, max_tokens=3)
    result = result.strip().upper()

    if "YES" in result:
        return "YES"
    if "NO" in result:
        return "NO"

    return "UNCLEAR"


if __name__ == "__main__":
    answer_1 = """
Force majeure refers to events beyond a party's reasonable control, such as fire, flood, earthquake, war, terrorism, riots, government actions, and similar events.
"""

    answer_2 = """
Force majeure means circumstances outside a party's control that may excuse delay or non-performance, including natural disasters, war, terrorism, and government action.
"""

    comparison = compare_meaning(answer_1, answer_2)

    print("\nSame meaning?")
    print(comparison)