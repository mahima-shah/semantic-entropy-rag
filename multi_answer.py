"""
multi_answer.py

Generates multiple independent answers for the same question using
the same retrieved context.

These multiple responses are later compared to estimate whether the
language model consistently interprets the retrieved evidence.
"""

from typing import Any

from llm import ask_llm
from rag import build_prompt, retrieve_chunks


# -----------------------------
# Multiple Answer Generation
# -----------------------------

def generate_multiple_answers(
    question: str,
    num_answers: int = 3
) -> dict[str, Any]:
    """
    Generate multiple answers for the same question.

    Each answer is produced independently while using the exact same
    retrieved context. The resulting answers are later compared by the
    semantic entropy layer to estimate confidence.

    Args:
        question:
            User question.

        num_answers:
            Number of independent answers to generate.

    Returns:
        Dictionary containing:
            - question
            - retrieved chunks
            - generated answers
    """

    # Retrieve supporting document chunks only once.
    chunks = retrieve_chunks(question)

    answers = []

    for i in range(num_answers):

        prompt = build_prompt(question, chunks)

        prompt += f"""

Generate answer version {i + 1}.

Requirements:
- Stay grounded in the retrieved context.
- Do not introduce outside knowledge.
- Answer naturally and independently.
"""

        answer = ask_llm(prompt)
        answers.append(answer)

    return {
        "question": question,
        "chunks": chunks,
        "answers": answers
    }


# -----------------------------
# Command-Line Entry Point
# -----------------------------

if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a question for multiple answers or type 'exit': "
        )

        if question.lower() == "exit":
            break

        result = generate_multiple_answers(question)

        print("\nQuestion:")
        print(result["question"])

        print("\nRetrieved Sources:")

        for i, chunk in enumerate(result["chunks"]):
            print(f"{i + 1}. {chunk['source']} | Chunk {chunk['chunk_index']}")

        print("\nGenerated Answers:")

        for i, answer in enumerate(result["answers"]):

            print("\n" + "=" * 80)
            print(f"Answer {i + 1}")
            print("=" * 80)

            print(answer)