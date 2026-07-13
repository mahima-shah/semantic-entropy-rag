"""
app.py

Streamlit interface for the RAG Reliability Checker.

The app:

1. Retrieves relevant legal-document chunks
2. Generates multiple answers at one fixed temperature
3. Groups answers by semantic meaning
4. Calculates normalized semantic entropy
5. Audits material contradictions
6. Routes to consensus finalization or recovery
7. Displays sources and debug details
8. Logs evaluation results to CSV
"""

import csv
import os
from datetime import datetime
from typing import Any

import streamlit as st

from pipeline import (
    run_reliability_pipeline
)


# -----------------------------
# Streamlit Page Setup
# -----------------------------

st.set_page_config(
    page_title="RAG Reliability Checker",
    page_icon="🔎",
    layout="wide"
)


# -----------------------------
# Helper Functions
# -----------------------------

def display_risk_badge(
    risk_label: str
) -> None:
    """
    Display semantic uncertainty.

    This is not labelled factual confidence because stable answers
    may still be consistently incorrect.
    """

    if risk_label == "Low":
        st.success(
            "Semantic uncertainty: Low"
        )

    elif risk_label == "Medium":
        st.warning(
            "Semantic uncertainty: Medium"
        )

    else:
        st.error(
            "Semantic uncertainty: High"
        )


def format_source_names(
    sources: list[dict[str, Any]]
) -> str:
    """
    Format source metadata for CSV logging.
    """

    return ", ".join(
        (
            f"{source['source']} "
            f"(Chunk {source['chunk_index']})"
        )
        for source in sources
    )


def save_result(
    result: dict[str, Any],
    category: str
) -> None:
    """
    Append one evaluation run to the CSV log.
    """

    os.makedirs(
        "results",
        exist_ok=True
    )

    filename = (
        "results/evaluation_results.csv"
    )

    file_exists = os.path.exists(
        filename
    )

    with open(
        filename,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Question",
                "Category",
                "Semantic Uncertainty",
                "Normalized Entropy",
                "Risk Reason",
                "Recovery Used",
                "Sample Count",
                "Sample Temperature",
                "Cluster Count",
                "Material Contradiction",
                "Answer",
                "Source Count",
                "Sources"
            ])

        writer.writerow([
            datetime.now().isoformat(
                timespec="seconds"
            ),
            result["question"],
            category,
            result["risk"]["label"],
            round(
                result["entropy"][
                    "normalized_entropy"
                ],
                4
            ),
            result["risk"]["reason"],
            result["recovery_used"],
            result["sample_count"],
            result[
                "sample_temperature"
            ],
            len(
                result["audit"][
                    "clusters"
                ]
            ),
            result["audit"][
                "material_contradiction"
            ],
            result["answer"].replace(
                "\n",
                " "
            ),
            len(result["sources"]),
            format_source_names(
                result["sources"]
            )
        ])


# -----------------------------
# UI
# -----------------------------

st.title(
    "RAG Reliability Checker"
)

st.write(
    "Ask a legal-document question. The app retrieves evidence, "
    "samples multiple answers, checks whether they agree in meaning, "
    "and uses a safer recovery path when contradictions appear."
)

question = st.text_input(
    "Ask a question"
)

category = st.selectbox(
    "Question category",
    [
        "Uncategorized",
        "Easy",
        "Ambiguous",
        "Impossible"
    ]
)

with st.expander(
    "Test settings"
):
    sample_count = st.selectbox(
        "Number of samples",
        [3, 5, 7],
        index=1
    )

    sample_temperature = st.slider(
        "Shared sampling temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help=(
            "Every sample in one run uses the same temperature. "
            "0.7 is a useful starting point."
        )
    )

    top_k = st.selectbox(
        "Number of retrieved chunks",
        [3, 4, 5, 6],
        index=1
    )

run_button = st.button(
    "Run reliability check",
    type="primary"
)


# -----------------------------
# Main App Flow
# -----------------------------

if (
    run_button
    and question.strip()
):
    try:
        with st.spinner(
            "Retrieving evidence, sampling answers, "
            "and auditing agreement..."
        ):
            result = (
                run_reliability_pipeline(
                    question=(
                        question.strip()
                    ),
                    sample_count=(
                        sample_count
                    ),
                    sample_temperature=(
                        sample_temperature
                    ),
                    top_k=top_k
                )
            )

        # -------------------------------------
        # Final Answer
        # -------------------------------------

        st.subheader(
            "Final Answer"
        )

        st.write(
            result["answer"]
        )

        # -------------------------------------
        # Reliability Signal
        # -------------------------------------

        st.subheader(
            "Reliability Signal"
        )

        display_risk_badge(
            result["risk"]["label"]
        )

        col_1, col_2, col_3 = (
            st.columns(3)
        )

        col_1.metric(
            "Normalized semantic entropy",
            (
                f"{result['entropy']['normalized_entropy']:.3f}"
            )
        )

        col_2.metric(
            "Meaning clusters",
            len(
                result["audit"][
                    "clusters"
                ]
            )
        )

        col_3.metric(
            "Recovery used",
            (
                "Yes"
                if result["recovery_used"]
                else "No"
            )
        )

        st.write(
            "**Why this route was selected:**"
        )

        st.write(
            result["risk"]["reason"]
        )

        st.caption(
            "Low semantic uncertainty means the sampled answers "
            "were stable in meaning. It does not prove that the "
            "answer is legally correct."
        )

        # -------------------------------------
        # Sources
        # -------------------------------------

        st.subheader(
            "Sources"
        )

        for index, source in enumerate(
            result["sources"]
        ):
            with st.container(
                border=True
            ):
                st.write(
                    f"**Source {index + 1}:** "
                    f"{source['source']} | "
                    f"Chunk "
                    f"{source['chunk_index']}"
                )

                snippet = (
                    source["text"][:700]
                    .replace("\n", " ")
                )

                if len(
                    source["text"]
                ) > 700:
                    snippet += "..."

                st.write(snippet)

        # -------------------------------------
        # Save Result
        # -------------------------------------

        save_result(
            result=result,
            category=category
        )

        st.success(
            "Result saved to "
            "results/evaluation_results.csv"
        )

        # -------------------------------------
        # Debug Details
        # -------------------------------------

        st.subheader(
            "Debug Details"
        )

        with st.expander(
            "Sampled Answers"
        ):
            for sample in result[
                "samples"
            ]:
                st.write(
                    f"### Sample "
                    f"{sample['sample_id']} "
                    f"(temperature="
                    f"{sample['temperature']})"
                )

                st.write(
                    sample["answer"]
                )

        with st.expander(
            "Semantic Clusters"
        ):
            for cluster in result[
                "audit"
            ]["clusters"]:
                st.write(
                    f"**Cluster "
                    f"{cluster.get('cluster_id')}:** "
                    f"{cluster.get('label')}"
                )

                sample_ids = (
                    cluster.get(
                        "sample_ids",
                        []
                    )
                )

                st.write(
                    "Samples: "
                    + ", ".join(
                        str(sample_id)
                        for sample_id
                        in sample_ids
                    )
                )

        with st.expander(
            "Contradiction Audit"
        ):
            st.json(
                result["audit"]
            )

        with st.expander(
            "Entropy Calculation"
        ):
            st.json(
                result["entropy"]
            )

        with st.expander(
            "Retrieved Chunks"
        ):
            for index, chunk in enumerate(
                result["sources"]
            ):
                st.write(
                    f"### Chunk "
                    f"{index + 1}"
                )

                st.write(
                    f"**Source:** "
                    f"{chunk['source']}"
                )

                st.write(
                    f"**Chunk Index:** "
                    f"{chunk['chunk_index']}"
                )

                st.write(
                    chunk["text"]
                )

    except Exception as error:
        st.error(
            f"Run failed: {error}"
        )


elif run_button:
    st.warning(
        "Please enter a question first."
    )