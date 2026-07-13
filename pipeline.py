"""
pipeline.py

Runs the complete RAG reliability workflow:

1. Retrieve evidence once
2. Generate repeated answers at one fixed temperature
3. Cluster the answers by semantic meaning
4. Calculate normalized semantic entropy
5. Audit material legal contradictions
6. Route to consensus finalization or evidence-grounded recovery
"""

from typing import Any

from entropy import (
    assess_risk,
    calculate_semantic_entropy,
    dominant_cluster
)
from multi_answer import (
    generate_multiple_answers
)
from nudging import (
    generate_consensus_answer,
    generate_recovery_answer
)
from rag import retrieve_chunks
from semantic_judge import judge_samples


def run_reliability_pipeline(
    question: str,
    sample_count: int = 5,
    sample_temperature: float = 0.7,
    top_k: int = 4
) -> dict[str, Any]:
    """
    Run the complete semantic uncertainty pipeline.
    """

    # -----------------------------------------
    # 1. Retrieve evidence once
    # -----------------------------------------

    chunks = retrieve_chunks(
        query=question,
        top_k=top_k
    )

    # -----------------------------------------
    # 2. Generate fixed-temperature samples
    # -----------------------------------------

    samples = generate_multiple_answers(
        question=question,
        chunks=chunks,
        sample_count=sample_count,
        temperature=sample_temperature
    )

    # -----------------------------------------
    # 3. Cluster and audit answers
    # -----------------------------------------

    audit = judge_samples(
        question=question,
        samples=samples
    )

    # -----------------------------------------
    # 4. Calculate cluster entropy
    # -----------------------------------------

    entropy_result = calculate_semantic_entropy(
        clusters=audit["clusters"],
        sample_count=len(samples)
    )

    # -----------------------------------------
    # 5. Determine risk route
    # -----------------------------------------

    risk = assess_risk(
        normalized_entropy=(
            entropy_result[
                "normalized_entropy"
            ]
        ),
        audit=audit
    )

    main_cluster = dominant_cluster(
        audit["clusters"]
    )

    # -----------------------------------------
    # 6. Produce final answer
    # -----------------------------------------

    if risk["route"] == "recover":
        final_answer = (
            generate_recovery_answer(
                question=question,
                chunks=chunks,
                samples=samples,
                audit=audit
            )
        )

        recovery_used = True

    else:
        if main_cluster:
            dominant_sample_ids = (
                main_cluster.get(
                    "sample_ids",
                    []
                )
            )
        else:
            dominant_sample_ids = [
                samples[0]["sample_id"]
            ]

        final_answer = (
            generate_consensus_answer(
                question=question,
                chunks=chunks,
                samples=samples,
                dominant_sample_ids=(
                    dominant_sample_ids
                ),
                medium_uncertainty=(
                    risk["route"]
                    == "qualify"
                )
            )
        )

        recovery_used = False

    return {
        "question": question,
        "answer": final_answer,
        "sources": chunks,
        "samples": samples,
        "audit": audit,
        "entropy": entropy_result,
        "risk": risk,
        "dominant_cluster": main_cluster,
        "recovery_used": recovery_used,
        "sample_count": sample_count,
        "sample_temperature": (
            sample_temperature
        )
    }