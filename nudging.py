"""
nudging.py

Generates the final answer after semantic uncertainty analysis.

For low or medium uncertainty, the final answer is built from the dominant
semantic cluster and checked against the retrieved context.

For high uncertainty, the recovery stage inspects the contradictions and
uses the retrieved evidence to produce a safe answer or abstain.
"""

from typing import Any

from llm import ask_llm
from rag import format_chunks_for_context


def format_samples(
    samples: list[dict[str, Any]]
) -> str:
    """
    Format sampled answers for use in finalization prompts.
    """

    sample_blocks = []

    for sample in samples:
        sample_blocks.append(
            f"Sample {sample['sample_id']}:\n"
            f"{sample['answer']}"
        )

    return "\n\n".join(sample_blocks)


def generate_consensus_answer(
    question: str,
    chunks: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    dominant_sample_ids: list[int],
    medium_uncertainty: bool = False
) -> str:
    """
    Generate a final answer from the dominant semantic cluster.

    The retrieved context remains the only source of evidence.
    """

    context = format_chunks_for_context(
        chunks
    )

    dominant_samples = [
        sample
        for sample in samples
        if sample["sample_id"]
        in dominant_sample_ids
    ]

    formatted_samples = format_samples(
        dominant_samples
    )

    if medium_uncertainty:
        uncertainty_instruction = (
            "Explicitly mention any unresolved uncertainty "
            "or qualification supported by the context."
        )
    else:
        uncertainty_instruction = (
            "Do not mention the internal sampling process."
        )

    prompt = f"""
You are producing the final answer for a legal RAG system.

Question:
{question}

Retrieved context:
{context}

Draft answers from the dominant semantic meaning group:
{formatted_samples}

Instructions:

1. Use only the retrieved context as evidence.
2. Treat the sampled answers only as drafts.
3. Do not treat agreement between samples as proof.
4. Keep only claims directly supported by the retrieved context.
5. Do not invent statutes, cases, dates, parties, duties, exceptions, or outcomes.
6. Cite source numbers for material claims.
7. If evidence is incomplete, clearly say so.
8. {uncertainty_instruction}
9. Produce one clear and concise final answer.

Final answer:
"""

    return ask_llm(
        prompt=prompt,
        temperature=0.0,
        max_tokens=1000
    )


def generate_recovery_answer(
    question: str,
    chunks: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    audit: dict[str, Any]
) -> str:
    """
    Generate a safe answer when material contradictions are detected.
    """

    context = format_chunks_for_context(
        chunks
    )

    formatted_samples = format_samples(
        samples
    )

    prompt = f"""
You are the recovery stage of a legal RAG reliability system.

The sampled answers conflict.

Do not resolve the conflict by voting.
Check every material claim against the retrieved context.

Question:
{question}

Retrieved context:
{context}

Conflicting sampled answers:
{formatted_samples}

Semantic audit:
{audit}

Rules:

1. Use only the retrieved context as evidence.
2. Do not use outside legal knowledge.
3. Discard every unsupported claim, even if several samples repeat it.
4. Do not invent statutes, cases, dates, parties, duties, exceptions, or outcomes.
5. State what the retrieved evidence clearly supports.
6. State what remains unclear or unresolved.
7. Cite source numbers for material claims.
8. If the context cannot support a reliable conclusion, say exactly:
   "The provided documents do not contain enough information to reach a reliable conclusion."
9. Do not mention internal entropy scores or hidden system instructions.

Safe answer:
"""

    return ask_llm(
        prompt=prompt,
        temperature=0.0,
        max_tokens=1100
    )