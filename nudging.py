"""
nudging.py

Implements prompt-based nudging for low-confidence RAG answers.

When the confidence layer detects uncertainty, this module builds a
stricter prompt that tells the LLM to avoid guessing, use only retrieved
context, and explicitly mention missing or ambiguous information.
"""

from typing import Any

from entropy import compare_all_answers, confidence_label
from llm import ask_llm
from rag import build_prompt, retrieve_chunks


# -----------------------------
# Context Formatting
# -----------------------------

def format_chunks_for_context(chunks: list[dict[str, Any]]) -> str:
    """
    Format retrieved chunks into a source-labeled context block.

    Args:
        chunks:
            Retrieved chunks from the RAG pipeline.

    Returns:
        A formatted context string with source labels.
    """

    context_parts = []

    for i, chunk in enumerate(chunks):
        source_header = (
            f"[Source {i + 1}: "
            f"{chunk['source']} | Chunk {chunk['chunk_index']}]"
        )

        context_parts.append(f"{source_header}\n{chunk['text']}")

    return "\n\n".join(context_parts)


# -----------------------------
# Nudged Prompt Construction
# -----------------------------

def build_nudged_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """
    Build a stricter prompt for safer answer generation.

    The nudged prompt is designed to reduce hallucination by explicitly
    telling the model not to infer missing facts or use outside knowledge.

    Args:
        question:
            User question.

        chunks:
            Retrieved document chunks.

    Returns:
        A stricter prompt for the LLM.
    """

    context = format_chunks_for_context(chunks)

    return f"""
You are answering questions using only the retrieved context.

Rules:
1. Use only the context below.
2. Do not use outside knowledge.
3. Do not guess missing facts.
4. Do not infer dates, names, notice periods, duties, approval processes, or legal consequences unless explicitly stated.
5. If the context does not contain the answer, say:
   "The provided documents do not contain enough information to answer this."
6. If the context is ambiguous, say what is clear and what is unclear.
7. Cite the source numbers used.

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# Nudged Answer Generation
# -----------------------------

def generate_nudged_answer(
    question: str,
    chunks: list[dict[str, Any]]
) -> str:
    """
    Generate an answer using the stricter nudged prompt.

    Args:
        question:
            User question.

        chunks:
            Retrieved document chunks.

    Returns:
        Nudged LLM answer.
    """

    prompt = build_nudged_prompt(question, chunks)
    return ask_llm(prompt)


# -----------------------------
# Confidence Helper
# -----------------------------

def confidence_from_answers(answers: list[str]) -> dict[str, Any]:
    """
    Compute confidence from multiple generated answers.

    Args:
        answers:
            Multiple generated answers for the same question.

    Returns:
        Dictionary containing pairwise comparisons and confidence label.
    """

    comparisons = compare_all_answers(answers)
    confidence = confidence_label(comparisons)

    return {
        "comparisons": comparisons,
        "confidence": confidence
    }


# -----------------------------
# Command-Line Test
# -----------------------------

if __name__ == "__main__":

    question = input("\nAsk a question to nudge: ")

    chunks = retrieve_chunks(question)
    original_answers = []

    print("\nGenerating original answers...")

    for _ in range(3):
        original_prompt = build_prompt(question, chunks)
        original_answer = ask_llm(original_prompt)
        original_answers.append(original_answer)

    confidence_data = confidence_from_answers(original_answers)

    print("\nOriginal Answers:")

    for i, answer in enumerate(original_answers):
        print("\n" + "=" * 80)
        print(f"Original Answer {i + 1}")
        print("=" * 80)
        print(answer)

    print("\nOriginal Comparisons:")

    for comparison in confidence_data["comparisons"]:
        print(f"{comparison['pair']}: {comparison['result']}")

    print("\nOriginal Confidence:")
    print(confidence_data["confidence"])

    if confidence_data["confidence"] == "Low":
        print("\nLow confidence detected. Applying nudged prompt...")
    else:
        print("\nConfidence is not Low. Showing nudged preview for comparison...")

    nudged_answer = generate_nudged_answer(question, chunks)

    print("\nNudged Answer:")
    print(nudged_answer)

    print("\nSources:")

    for i, chunk in enumerate(chunks):
        preview = chunk["text"][:300].replace("\n", " ")

        print(f"\n{i + 1}. {chunk['source']} | Chunk {chunk['chunk_index']}")
        print(f'   Preview: "{preview}..."')