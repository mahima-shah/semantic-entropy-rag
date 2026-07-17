"""
pipeline.py

Runs both reliability experiments:

1. Semantic uncertainty test using repeated samples at one temperature.
2. Temperature sensitivity test using a sweep of different temperatures.
"""

from typing import Any

from entropy import (
    assess_risk,
    calculate_semantic_entropy,
    dominant_cluster
)
from multi_answer import (
    generate_fixed_temperature_answers,
    generate_temperature_sweep_answers
)
from nudging import (
    generate_consensus_answer,
    generate_recovery_answer
)
from rag import retrieve_chunks
from semantic_judge import (
    judge_samples,
    judge_temperature_sensitivity
)


def run_reliability_pipeline(
    question: str,
    sample_count: int = 5,
    sample_temperature: float = 0.7,
    temperature_sweep: list[float] | None = None,
    top_k: int = 4,
    run_temperature_test: bool = True
) -> dict[str, Any]:
    """
    Run semantic uncertainty and optional temperature sensitivity tests.
    """

    # -----------------------------------------
    # 1. Retrieve evidence once
    # -----------------------------------------

    chunks = retrieve_chunks(
        query=question,
        top_k=top_k
    )

    # -----------------------------------------
    # 2. Fixed-temperature semantic test
    # -----------------------------------------

    semantic_samples = (
        generate_fixed_temperature_answers(
            question=question,
            chunks=chunks,
            sample_count=sample_count,
            temperature=sample_temperature
        )
    )

    semantic_audit = judge_samples(
        question=question,
        samples=semantic_samples
    )

    entropy_result = (
        calculate_semantic_entropy(
            clusters=semantic_audit[
                "clusters"
            ],
            sample_count=len(
                semantic_samples
            )
        )
    )

    risk = assess_risk(
        normalized_entropy=(
            entropy_result[
                "normalized_entropy"
            ]
        ),
        audit=semantic_audit
    )

    main_cluster = dominant_cluster(
        semantic_audit["clusters"]
    )

    # -----------------------------------------
    # 3. Final answer routing
    # -----------------------------------------

    if risk["route"] == "recover":
        final_answer = (
            generate_recovery_answer(
                question=question,
                chunks=chunks,
                samples=semantic_samples,
                audit=semantic_audit
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
                semantic_samples[0][
                    "sample_id"
                ]
            ]

        final_answer = (
            generate_consensus_answer(
                question=question,
                chunks=chunks,
                samples=semantic_samples,
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

    # -----------------------------------------
    # 4. Temperature sensitivity test
    # -----------------------------------------

    temperature_samples = []
    temperature_audit = None

    if run_temperature_test:
        temperature_samples = (
            generate_temperature_sweep_answers(
                question=question,
                chunks=chunks,
                temperatures=temperature_sweep
            )
        )

        temperature_audit = (
            judge_temperature_sensitivity(
                question=question,
                samples=temperature_samples
            )
        )

    return {
        "question": question,
        "answer": final_answer,
        "sources": chunks,

        "semantic_test": {
            "samples": semantic_samples,
            "audit": semantic_audit,
            "entropy": entropy_result,
            "risk": risk,
            "dominant_cluster": (
                main_cluster
            ),
            "sample_count": sample_count,
            "sample_temperature": (
                sample_temperature
            )
        },

        "temperature_test": {
            "enabled": run_temperature_test,
            "samples": temperature_samples,
            "audit": temperature_audit,
            "temperatures": (
                temperature_sweep
            )
        },

        "recovery_used": recovery_used
    }