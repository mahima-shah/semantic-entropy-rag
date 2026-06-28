import streamlit as st

from rag import answer_question
from multi_answer import generate_multiple_answers
from entropy import compare_all_answers, confidence_label
from nudging import generate_nudged_answer


st.set_page_config(
    page_title="RAG Reliability Checker",
    page_icon="🔎",
    layout="wide"
)

st.title("RAG Reliability Checker")
st.write("Ask a question and see the answer, sources, confidence score, and nudged response.")

question = st.text_input("Ask a question")

run_button = st.button("Run")

def confidence_reason(confidence):
    if confidence == "High":
        return "All generated answers agreed in meaning."
    elif confidence == "Medium":
        return "Most generated answers agreed, but one comparison disagreed."
    else:
        return "Multiple generated answers disagreed, so the answer may be unreliable."

def confidence_badge(confidence):
    if confidence == "High":
        st.success("Confidence: High")
    elif confidence == "Medium":
        st.warning("Confidence: Medium")
    else:
        st.error("Confidence: Low")

if run_button and question:
    with st.spinner("Running RAG pipeline..."):
        rag_result = answer_question(question)

    st.subheader("Answer")
    st.write(rag_result["answer"])

    with st.spinner("Generating multiple answers for confidence scoring..."):
        multi_result = generate_multiple_answers(question)
        answers = multi_result["answers"]
        comparisons = compare_all_answers(answers)
        confidence = confidence_label(comparisons)

    st.subheader("Confidence")
    confidence_badge(confidence)

    st.write("**Reason:**")
    st.write(confidence_reason(confidence))

    st.subheader("Sources")
    for i, source in enumerate(rag_result["sources"]):
        with st.container(border=True):
            st.write(f"**Source {i+1}:** {source['source']} | Chunk {source['chunk_index']}")
            snippet = source["text"][:700].replace("\n", " ")
            st.write(snippet + "...")

    nudged_answer = None

    if confidence == "Low":
        with st.spinner("Low confidence detected. Applying nudged prompt..."):
            nudged_answer = generate_nudged_answer(question, rag_result["sources"])

        st.subheader("Nudged Answer")
        st.write(nudged_answer)
    else:
        with st.expander("Nudged Answer Preview"):
            nudged_answer = generate_nudged_answer(question, rag_result["sources"])
            st.write(nudged_answer)

    st.subheader("Debug Details")

    with st.expander("Retrieved Chunks"):
        for i, chunk in enumerate(rag_result["sources"]):
            st.write(f"### Chunk {i+1}")
            st.write(f"**Source:** {chunk['source']}")
            st.write(f"**Chunk Index:** {chunk['chunk_index']}")
            st.write(chunk["text"])

    with st.expander("Generated Answers"):
        for i, answer in enumerate(answers):
            st.write(f"### Answer {i+1}")
            st.write(answer)

    with st.expander("Meaning Comparisons"):
        for item in comparisons:
            st.write(f"**{item['pair']}:** {item['result']}")

    with st.expander("Nudged Answer"):
        st.write(nudged_answer)

elif run_button and not question:
    st.warning("Please enter a question first.")