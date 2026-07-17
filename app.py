"""
app.py

Streamlit interface for the RAG Reliability Checker.

The app runs two separate reliability tests:

1. Semantic uncertainty test
   Generates multiple answers at one fixed temperature and checks whether
   they agree in meaning.

2. Temperature sensitivity test
   Generates answers at different temperatures and checks whether changing
   the temperature changes the legal conclusion.

The app also:
- retrieves document evidence
- generates a final answer
- applies recovery when semantic contradictions are detected
- displays sources and debug details
- logs each run to a CSV file
"""

import csv
import os
from datetime import datetime
from typing import Any

import streamlit as st

from pipeline import run_reliability_pipeline


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

def display_semantic_risk_badge(
    risk_label: str
) -> None:
    """
    Display the result of the fixed-temperature semantic uncertainty test.
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


def display_temperature_badge(
    sensitivity_level: str
) -> None:
    """
    Display the result of the temperature sensitivity test.
    """

    if sensitivity_level == "Low":
        st.success(
            "Temperature sensitivity: Low"
        )

    elif sensitivity_level == "Medium":
        st.warning(
            "Temperature sensitivity: Medium"
        )

    else:
        st.error(
            "Temperature sensitivity: High"
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
    Append one evaluation run to results/evaluation_results.csv.
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

    semantic_result = result[
        "semantic_test"
    ]

    temperature_result = result[
        "temperature_test"
    ]

    temperature_audit = (
        temperature_result.get("audit")
        or {}
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
                "Semantic Risk Reason",
                "Recovery Used",
                "Semantic Sample Count",
                "Semantic Sample Temperature",
                "Semantic Cluster Count",
                "Material Contradiction",
                "Temperature Test Enabled",
                "Temperature Sensitivity",
                "Stable Across Temperatures",
                "Highest Risk Temperature",
                "Final Answer",
                "Source Count",
                "Sources"
            ])

        writer.writerow([
            datetime.now().isoformat(
                timespec="seconds"
            ),
            result["question"],
            category,
            semantic_result[
                "risk"
            ]["label"],
            round(
                semantic_result[
                    "entropy"
                ]["normalized_entropy"],
                4
            ),
            semantic_result[
                "risk"
            ]["reason"],
            result[
                "recovery_used"
            ],
            semantic_result[
                "sample_count"
            ],
            semantic_result[
                "sample_temperature"
            ],
            len(
                semantic_result[
                    "audit"
                ]["clusters"]
            ),
            semantic_result[
                "audit"
            ]["material_contradiction"],
            temperature_result[
                "enabled"
            ],
            temperature_audit.get(
                "sensitivity_level",
                ""
            ),
            temperature_audit.get(
                "stable_across_temperatures",
                ""
            ),
            temperature_audit.get(
                "highest_risk_temperature",
                ""
            ),
            result[
                "answer"
            ].replace(
                "\n",
                " "
            ),
            len(
                result[
                    "sources"
                ]
            ),
            format_source_names(
                result[
                    "sources"
                ]
            )
        ])


# -----------------------------
# UI
# -----------------------------

st.title(
    "RAG Reliability Checker"
)

st.write(
    "Ask a legal-document question. The app runs two reliability tests: "
    "one checks whether repeated answers agree at the same temperature, "
    "and the other checks whether the answer changes across temperatures."
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


# -----------------------------
# Test Settings
# -----------------------------

with st.expander(
    "Test settings"
):

    st.write(
        "**Semantic uncertainty test**"
    )

    sample_count = st.selectbox(
        "Number of fixed-temperature samples",
        [3, 5, 7],
        index=1,
        help=(
            "The same question is generated multiple times "
            "using the same evidence and temperature."
        )
    )

    sample_temperature = st.slider(
        "Shared sampling temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help=(
            "All semantic uncertainty samples use this same temperature."
        )
    )

    st.divider()

    st.write(
        "**Temperature sensitivity test**"
    )

    run_temperature_test = st.checkbox(
        "Also run temperature sensitivity test",
        value=True,
        help=(
            "Generates one answer at several temperatures "
            "to see whether the legal conclusion changes."
        )
    )

    st.caption(
        "Temperature sweep: 0.0, 0.3, 0.6 and 0.9"
    )

    st.divider()

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
            "Retrieving evidence, generating samples, "
            "and running both reliability tests..."
        ):

            result = run_reliability_pipeline(
                question=question.strip(),
                sample_count=sample_count,
                sample_temperature=sample_temperature,
                temperature_sweep=[
                    0.0,
                    0.3,
                    0.6,
                    0.9
                ],
                top_k=top_k,
                run_temperature_test=(
                    run_temperature_test
                )
            )

        semantic_result = result[
            "semantic_test"
        ]

        temperature_result = result[
            "temperature_test"
        ]


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
        # Test 1: Semantic Uncertainty
        # -------------------------------------

        st.subheader(
            "1. Semantic Uncertainty Test"
        )

        st.caption(
            "This test generates multiple answers using the same "
            "temperature, prompt and retrieved evidence."
        )

        display_semantic_risk_badge(
            semantic_result[
                "risk"
            ]["label"]
        )

        semantic_col_1, semantic_col_2, semantic_col_3 = (
            st.columns(3)
        )

        semantic_col_1.metric(
            "Normalized semantic entropy",
            (
                f"{semantic_result['entropy']['normalized_entropy']:.3f}"
            )
        )

        semantic_col_2.metric(
            "Meaning clusters",
            len(
                semantic_result[
                    "audit"
                ]["clusters"]
            )
        )

        semantic_col_3.metric(
            "Recovery used",
            (
                "Yes"
                if result[
                    "recovery_used"
                ]
                else "No"
            )
        )

        st.write(
            "**Why this route was selected:**"
        )

        st.write(
            semantic_result[
                "risk"
            ]["reason"]
        )

        st.caption(
            "Low semantic uncertainty means the sampled answers "
            "were stable in meaning. It does not prove that the "
            "answer is legally correct."
        )


        # -------------------------------------
        # Test 2: Temperature Sensitivity
        # -------------------------------------

        if temperature_result[
            "enabled"
        ]:

            st.subheader(
                "2. Temperature Sensitivity Test"
            )

            st.caption(
                "This test generates one answer at each temperature "
                "to see whether changing the model setting changes "
                "the legal conclusion."
            )

            temperature_audit = (
                temperature_result[
                    "audit"
                ]
                or {}
            )

            sensitivity_level = (
                temperature_audit.get(
                    "sensitivity_level",
                    "Unknown"
                )
            )

            display_temperature_badge(
                sensitivity_level
            )

            temperature_col_1, temperature_col_2 = (
                st.columns(2)
            )

            stable_across_temperatures = (
                temperature_audit.get(
                    "stable_across_temperatures",
                    False
                )
            )

            temperature_col_1.metric(
                "Stable conclusion",
                (
                    "Yes"
                    if stable_across_temperatures
                    else "No"
                )
            )

            highest_risk_temperature = (
                temperature_audit.get(
                    "highest_risk_temperature"
                )
            )

            temperature_col_2.metric(
                "Highest-risk temperature",
                (
                    str(
                        highest_risk_temperature
                    )
                    if highest_risk_temperature
                    is not None
                    else "None"
                )
            )

            st.write(
                "**Temperature test summary:**"
            )

            st.write(
                temperature_audit.get(
                    "summary",
                    "No temperature sensitivity summary was returned."
                )
            )

            temperature_changes = (
                temperature_audit.get(
                    "changes",
                    []
                )
            )

            if temperature_changes:

                st.write(
                    "**Changes detected:**"
                )

                for change in temperature_changes:

                    with st.container(
                        border=True
                    ):

                        change_temperature = (
                            change.get(
                                "temperature",
                                "Unknown"
                            )
                        )

                        change_type = (
                            change.get(
                                "type",
                                "change"
                            )
                            .replace(
                                "_",
                                " "
                            )
                            .title()
                        )

                        st.write(
                            f"**Temperature "
                            f"{change_temperature}: "
                            f"{change_type}**"
                        )

                        st.write(
                            change.get(
                                "description",
                                "No description provided."
                            )
                        )

            else:

                st.info(
                    "No material changes were detected across temperatures."
                )


        # -------------------------------------
        # Sources
        # -------------------------------------

        st.subheader(
            "Sources"
        )

        for index, source in enumerate(
            result[
                "sources"
            ]
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
                    source[
                        "text"
                    ][:700]
                    .replace(
                        "\n",
                        " "
                    )
                )

                if len(
                    source[
                        "text"
                    ]
                ) > 700:

                    snippet += "..."

                st.write(
                    snippet
                )


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


        # Fixed-temperature samples

        with st.expander(
            "Semantic Uncertainty Samples"
        ):

            for sample in semantic_result[
                "samples"
            ]:

                st.write(
                    f"### Sample "
                    f"{sample['sample_id']} "
                    f"(temperature="
                    f"{sample['temperature']})"
                )

                st.write(
                    sample[
                        "answer"
                    ]
                )


        # Semantic clusters

        with st.expander(
            "Semantic Clusters"
        ):

            for cluster in semantic_result[
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
                        str(
                            sample_id
                        )
                        for sample_id
                        in sample_ids
                    )
                )


        # Contradiction audit

        with st.expander(
            "Contradiction Audit"
        ):

            st.json(
                semantic_result[
                    "audit"
                ]
            )


        # Entropy calculation

        with st.expander(
            "Entropy Calculation"
        ):

            st.json(
                semantic_result[
                    "entropy"
                ]
            )


        # Temperature samples and audit

        if temperature_result[
            "enabled"
        ]:

            with st.expander(
                "Temperature Sweep Samples"
            ):

                for sample in temperature_result[
                    "samples"
                ]:

                    st.write(
                        f"### Temperature "
                        f"{sample['temperature']}"
                    )

                    st.write(
                        sample[
                            "answer"
                        ]
                    )

            with st.expander(
                "Temperature Sensitivity Audit"
            ):

                st.json(
                    temperature_result[
                        "audit"
                    ]
                )


        # Retrieved chunks

        with st.expander(
            "Retrieved Chunks"
        ):

            for index, chunk in enumerate(
                result[
                    "sources"
                ]
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
                    chunk[
                        "text"
                    ]
                )


    except Exception as error:

        st.error(
            f"Run failed: {error}"
        )


elif run_button:

    st.warning(
        "Please enter a question first."
    )