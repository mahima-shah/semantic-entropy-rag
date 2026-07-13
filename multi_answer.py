"""
multi_answer.py

Generates multiple answers for the same question using the same retrieved
context, prompt, model, and temperature.

Keeping the temperature fixed means the test measures variation within one
sampling distribution rather than variation caused by changing settings.
"""

from typing import Any

from llm import ask_llm
from rag import build_prompt


DEFAULT_SAMPLE_COUNT = 5
DEFAULT_SAMPLE_TEMPERATURE = 0.7


def generate_multiple_answers(
    question: str,
    chunks: list[dict[str, Any]],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    temperature: float = DEFAULT_SAMPLE_TEMPERATURE
) -> list[dict[str, Any]]:
    """
    Generate repeated answers using one fixed temperature.

    Args:
        question:
            Original user question.

        chunks:
            Retrieved chunks that remain fixed for all samples.

        sample_count:
            Number of answers to generate.

        temperature:
            Shared temperature used for every answer.

    Returns:
        List containing each generated answer and its metadata.
    """

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    prompt = build_prompt(
        question,
        chunks
    )

    samples = []

    for index in range(sample_count):
        answer = ask_llm(
            prompt=prompt,
            temperature=temperature,
            max_tokens=900
        )

        samples.append({
            "sample_id": index + 1,
            "temperature": temperature,
            "answer": answer
        })

    return samples


if __name__ == "__main__":
    from rag import retrieve_chunks

    question = input(
        "\nAsk a question for semantic consistency testing: "
    )

    chunks = retrieve_chunks(question)

    samples = generate_multiple_answers(
        question=question,
        chunks=chunks,
        sample_count=5,
        temperature=0.7
    )

    for sample in samples:
        print("\n" + "=" * 80)
        print(
            f"Sample {sample['sample_id']} "
            f"(temperature={sample['temperature']})"
        )
        print("=" * 80)
        print(sample["answer"])