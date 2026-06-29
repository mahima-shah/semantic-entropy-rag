"""
entropy.py

Implements a lightweight semantic entropy-inspired confidence layer.

Instead of relying on a single generated answer, the system generates
multiple answers and compares their meanings. The level of agreement
between the answers is then used to estimate confidence.
"""

from compare_answers import compare_meaning


# -----------------------------
# Pairwise Meaning Comparison
# -----------------------------

def compare_all_answers(answers: list[str]) -> list[dict]:
    """
    Compare every pair of generated answers.

    Each pair is evaluated using an LLM that determines whether the
    two answers communicate the same meaning.

    Args:
        answers:
            List of generated answers.

    Returns:
        List of comparison results.
    """

    comparisons = []

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):

            result = compare_meaning(
                answers[i],
                answers[j]
            )

            comparisons.append({
                "pair": f"A{i + 1} vs A{j + 1}",
                "result": result
            })

    return comparisons


# -----------------------------
# Confidence Estimation
# -----------------------------

# NOTE:
# This is a lightweight approximation of semantic entropy.
# Rather than computing probability distributions over semantic
# clusters as described in the original research, this project
# estimates confidence using pairwise agreement between multiple
# independently generated answers.

def confidence_label(comparisons: list[dict]) -> str:
    """
    Estimate confidence based on semantic agreement.

    Confidence is determined by counting how many pairwise comparisons
    disagree in meaning.

    Rules:
        High:
            No disagreements.

        Medium:
            One disagreement.

        Low:
            Two or more disagreements.

    Args:
        comparisons:
            Output from compare_all_answers().

    Returns:
        Confidence label ("High", "Medium", or "Low").
    """

    no_count = sum(
        1
        for comparison in comparisons
        if comparison["result"] == "NO"
    )

    if no_count == 0:
        return "High"

    elif no_count == 1:
        return "Medium"

    else:
        return "Low"


# -----------------------------
# Command-Line Example
# -----------------------------

if __name__ == "__main__":

    answers = [
        "Force majeure means events beyond a party's control, such as natural disasters or war.",

        "Force majeure refers to uncontrollable events that excuse non-performance, like floods, earthquakes, or terrorism.",

        "Force majeure means the notice period is 30 days."
    ]

    comparisons = compare_all_answers(answers)
    confidence = confidence_label(comparisons)

    print("Comparisons:")

    for comparison in comparisons:
        print(f"{comparison['pair']}: {comparison['result']}")

    print("\nConfidence:")
    print(confidence)