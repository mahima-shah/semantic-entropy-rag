from compare_answers import compare_meaning


def compare_all_answers(answers):
    comparisons = []

    for i in range(len(answers)):
        for j in range(i + 1, len(answers)):
            result = compare_meaning(answers[i], answers[j])
            comparisons.append({
                "pair": f"A{i+1} vs A{j+1}",
                "result": result
            })

    return comparisons


def confidence_label(comparisons):
    no_count = sum(1 for item in comparisons if item["result"] == "NO")

    if no_count == 0:
        return "High"
    elif no_count == 1:
        return "Medium"
    else:
        return "Low"


if __name__ == "__main__":
    answers = [
        "Force majeure means events beyond a party's control, such as natural disasters or war.",
        "Force majeure refers to uncontrollable events that excuse non-performance, like floods, earthquakes, or terrorism.",
        "Force majeure means the notice period is 30 days."
    ]

    comparisons = compare_all_answers(answers)
    confidence = confidence_label(comparisons)

    print("Comparisons:")
    for item in comparisons:
        print(f"{item['pair']}: {item['result']}")

    print("\nConfidence:")
    print(confidence)