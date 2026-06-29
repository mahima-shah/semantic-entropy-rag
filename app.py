"""
app.py

Streamlit interface for the RAG Reliability Checker.

The app lets a user:
1. Ask a question
2. View the RAG-generated answer
3. See the confidence score
4. Inspect retrieved sources
5. View the nudged answer
6. Expand debug details for evaluation
7. Automatically log each run to a CSV file
"""

import csv
import os
from datetime import datetime
from typing import Any

import streamlit as st

from entropy import compare_all_answers, confidence_label
from multi_answer import generate_multiple_answers
from nudging import generate_nudged_answer
from rag import answer_question


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

def confidence_reason(confidence: str) -> str:
    """
    Convert a confidence label into a human-readable explanation.
    """

    if confidence == "High":
        return "All generated answers agreed in meaning."

    if confidence == "Medium":
        return "Most generated answers agreed, but one comparison disagreed."

    return "Multiple generated answers disagreed, so the answer may be unreliable."


def display_confidence_badge(confidence: str) -> None:
    """
    Display confidence using Streamlit's status components.
    """

    if confidence == "High":
        st.success("Confidence: High")

    elif confidence == "Medium":
        st.warning("Confidence: Medium")

    else:
        st.error("Confidence: Low")


def format_source_names(sources: list[dict[str, Any]]) -> str:
    """
    Format source metadata for CSV logging.
    """

    return ", ".join(
        f"{source['source']} (Chunk {source['chunk_index']})"
        for source in sources
    )


def save_result(
    question: str,
    category: str,
    confidence: str,
    reason: str,
    nudged_used: bool,
    answer: str,
    sources: list[dict[str, Any]]
) -> None:
    """
    Append one evaluation run to results/evaluation_results.csv.

    This creates a lightweight experiment log so evaluation does not
    need to be copied manually from the Streamlit interface.
    """

    os.makedirs("results", exist_ok=True)

    filename = "results/evaluation_results.csv"
    file_exists = os.path.exists(filename)

    with open(filename, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "Timestamp",
                "Question",
                "Category",
                "Confidence",
                "Reason",
                "Nudged Used",
                "Answer",
                "Source Count",
                "Sources"
            ])

        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            question,
            category,
            confidence,
            reason,
            nudged_used,
            answer.replace("\n", " "),
            len(sources),
            format_source_names(sources)
        ])


# -----------------------------
# UI
# -----------------------------

st.title("RAG Reliability Checker")
st.write(
    "Ask a question and see the answer, sources, confidence score, "
    "and nudged response."
)

question = st.text_input("Ask a question")

category = st.selectbox(
    "Question category",
    ["Uncategorized", "Easy", "Ambiguous", "Impossible"]
)

run_button = st.button("Run")


# -----------------------------
# Main App Flow
# -----------------------------

if run_button and question:

    # 1. Run the standard RAG pipeline.
    with st.spinner("Running RAG pipeline..."):
        rag_result = answer_question(question)

    st.subheader("Answer")
    st.write(rag_result["answer"])

    # 2. Generate multiple answers and compare them for confidence scoring.
    with st.spinner("Generating multiple answers for confidence scoring..."):
        multi_result = generate_multiple_answers(question)
        answers = multi_result["answers"]
        comparisons = compare_all_answers(answers)
        confidence = confidence_label(comparisons)

    reason = confidence_reason(confidence)

    st.subheader("Confidence")
    display_confidence_badge(confidence)

    st.write("**Reason:**")
    st.write(reason)

    # 3. Display retrieved sources.
    st.subheader("Sources")

    for i, source in enumerate(rag_result["sources"]):
        with st.container(border=True):
            st.write(
                f"**Source {i + 1}:** "
                f"{source['source']} | Chunk {source['chunk_index']}"
            )

            snippet = source["text"][:700].replace("\n", " ")
            st.write(snippet + "...")

    # 4. Generate nudged answer.
    # In production, this is used only when confidence is Low.
    # In the UI, a preview is also shown for High/Medium cases.
    nudged_used = confidence == "Low"

    if nudged_used:
        with st.spinner("Low confidence detected. Applying nudged prompt..."):
            nudged_answer = generate_nudged_answer(
                question,
                rag_result["sources"]
            )

        st.subheader("Nudged Answer")
        st.write(nudged_answer)

    else:
        with st.expander("Nudged Answer Preview"):
            nudged_answer = generate_nudged_answer(
                question,
                rag_result["sources"]
            )
            st.write(nudged_answer)

    # 5. Save run to evaluation CSV.
    save_result(
        question=question,
        category=category,
        confidence=confidence,
        reason=reason,
        nudged_used=nudged_used,
        answer=rag_result["answer"],
        sources=rag_result["sources"]
    )

    st.success("Result saved to results/evaluation_results.csv")

    # 6. Debug details for evaluation and transparency.
    st.subheader("Debug Details")

    with st.expander("Retrieved Chunks"):
        for i, chunk in enumerate(rag_result["sources"]):
            st.write(f"### Chunk {i + 1}")
            st.write(f"**Source:** {chunk['source']}")
            st.write(f"**Chunk Index:** {chunk['chunk_index']}")
            st.write(chunk["text"])

    with st.expander("Generated Answers"):
        for i, answer in enumerate(answers):
            st.write(f"### Answer {i + 1}")
            st.write(answer)

    with st.expander("Meaning Comparisons"):
        for comparison in comparisons:
            st.write(f"**{comparison['pair']}:** {comparison['result']}")

    with st.expander("Nudged Answer"):
        st.write(nudged_answer)


elif run_button and not question:
    st.warning("Please enter a question first.")