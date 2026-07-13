"""
entropy.py

Calculates a frequency-based semantic entropy approximation from the
semantic meaning clusters produced by semantic_judge.py.

This is not the full probability-weighted semantic entropy calculation.
It is a practical prototype using the frequency of sampled meanings.
"""

import math
from typing import Any


def calculate_semantic_entropy(
    clusters: list[dict[str, Any]],
    sample_count: int
) -> dict[str, float]:
    """
    Calculate raw and normalized entropy from cluster frequencies.

    Args:
        clusters:
            Semantic answer clusters returned by the judge.

        sample_count:
            Total number of generated answers.

    Returns:
        Raw entropy and normalized entropy.
    """

    if sample_count < 2:
        return {
            "raw_entropy": 0.0,
            "normalized_entropy": 0.0
        }

    cluster_sizes = [
        len(cluster.get("sample_ids", []))
        for cluster in clusters
        if cluster.get("sample_ids")
    ]

    if not cluster_sizes:
        return {
            "raw_entropy": 0.0,
            "normalized_entropy": 0.0
        }

    raw_entropy = 0.0

    for cluster_size in cluster_sizes:
        probability = (
            cluster_size / sample_count
        )

        if probability > 0:
            raw_entropy -= (
                probability
                * math.log(probability)
            )

    maximum_entropy = math.log(
        sample_count
    )

    if maximum_entropy == 0:
        normalized_entropy = 0.0
    else:
        normalized_entropy = (
            raw_entropy / maximum_entropy
        )

    normalized_entropy = min(
        max(normalized_entropy, 0.0),
        1.0
    )

    return {
        "raw_entropy": raw_entropy,
        "normalized_entropy": normalized_entropy
    }


def dominant_cluster(
    clusters: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """
    Return the semantic cluster containing the most answers.
    """

    if not clusters:
        return None

    return max(
        clusters,
        key=lambda cluster: len(
            cluster.get("sample_ids", [])
        )
    )


def assess_risk(
    normalized_entropy: float,
    audit: dict[str, Any]
) -> dict[str, str]:
    """
    Convert entropy and contradiction signals into a risk route.

    Material legal disagreements override the numeric entropy score.
    Citation-format differences alone do not trigger recovery.
    """

    hard_override_reasons = []

    if audit.get("material_contradiction"):
        hard_override_reasons.append(
            "Material contradiction detected."
        )

    if audit.get("jurisdiction_variance"):
        hard_override_reasons.append(
            "The answers used different jurisdictions."
        )

    if audit.get("unsupported_claim_risk"):
        hard_override_reasons.append(
            "The answers may contain unsupported factual claims."
        )

    if (
        hard_override_reasons
        or normalized_entropy > 0.50
    ):
        reason = " ".join(
            hard_override_reasons
        )

        if not reason:
            reason = (
                "The answers were distributed across "
                "multiple semantic meanings."
            )

        return {
            "label": "High",
            "route": "recover",
            "reason": reason
        }

    if normalized_entropy > 0.20:
        reason = (
            "Most answers agreed, but some semantic "
            "variation remained."
        )

        if audit.get("citation_variance"):
            reason += (
                " Citation references also varied."
            )

        return {
            "label": "Medium",
            "route": "qualify",
            "reason": reason
        }

    if audit.get("citation_variance"):
        reason = (
            "The sampled answers were stable in meaning, "
            "but their citation labels or formatting varied."
        )
    else:
        reason = (
            "The sampled answers were stable in meaning."
        )

    return {
        "label": "Low",
        "route": "proceed",
        "reason": reason
    }