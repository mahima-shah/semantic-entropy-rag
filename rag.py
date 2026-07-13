"""
rag.py

Core Retrieval-Augmented Generation pipeline.

This file:
1. Retrieves relevant chunks from ChromaDB
2. Builds a context-grounded prompt
3. Sends the prompt to the LLM
4. Returns the answer with source metadata
"""

import os
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer

from llm import ask_llm


# Prevent tokenizer parallelism warnings when used inside Streamlit.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# -----------------------------
# Configuration
# -----------------------------

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "legal_docs"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# Load the embedding model once at startup.
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")


# -----------------------------
# Retrieval
# -----------------------------

def retrieve_chunks(query: str, top_k: int = 4) -> list[dict[str, Any]]:
    """
    Retrieve the most relevant document chunks for a user question.

    Args:
        query:
            User question.

        top_k:
            Number of chunks to retrieve.

    Returns:
        A list of retrieved chunks with text and metadata.
    """

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    chunks = []

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        chunks.append({
            "text": document,
            "source": metadata["source"],
            "chunk_index": metadata["chunk_index"]
        })

    return chunks

def format_chunks_for_context(
    chunks: list[dict[str, Any]]
) -> str:
    """
    Format retrieved chunks into a source-labeled context block.
    """

    context_parts = []

    for i, chunk in enumerate(chunks):
        source_header = (
            f"[Source {i + 1}: "
            f"{chunk['source']} | "
            f"Chunk {chunk['chunk_index']}]"
        )

        context_parts.append(
            f"{source_header}\n"
            f"{chunk['text']}"
        )

    return "\n\n".join(
        context_parts
    )
# -----------------------------
# Prompt Construction
# -----------------------------

def build_prompt(
    question: str,
    chunks: list[dict[str, Any]]
) -> str:
    """
    Build the evidence-grounded prompt shared by all sampled answers.
    """

    context = format_chunks_for_context(
        chunks
    )

    return f"""
You are a legal-document question-answering assistant.

Use only the retrieved context below.

Rules:

1. Do not use outside knowledge.
2. Do not invent statutes, cases, dates, parties, duties, exceptions, or outcomes.
3. Distinguish what the document expressly states from interpretation.
4. If the answer is missing, say:
   "The provided documents do not contain enough information to answer this."
5. If the context is ambiguous, explain what is clear and what remains unclear.
6. Cite source numbers for material claims.
7. Keep the answer focused on the question.

Retrieved context:
{context}

Question:
{question}

Answer:
"""

def generate_final_answer(
    question: str,
    chunks: list[dict[str, Any]]
) -> str:
    """
    Generate the stable answer shown to the user.

    The final answer uses temperature 0.0 because it is not part of
    uncertainty sampling.
    """

    prompt = build_prompt(
        question,
        chunks
    )

    return ask_llm(
        prompt=prompt,
        temperature=0.0
    )

# -----------------------------
# End-to-End RAG
# -----------------------------

def answer_question(
    question: str
) -> dict[str, Any]:
    """
    Retrieve evidence once and generate a stable answer.
    """

    chunks = retrieve_chunks(question)

    answer = generate_final_answer(
        question,
        chunks
    )

    return {
        "question": question,
        "answer": answer,
        "sources": chunks
    }


# -----------------------------
# Command-Line Entry Point
# -----------------------------

if __name__ == "__main__":
    while True:
        question = input("\nAsk a RAG question or type 'exit': ")

        if question.lower() == "exit":
            break

        result = answer_question(question)

        print("\nAnswer:")
        print(result["answer"])

        print("\nSources:")
        for i, source in enumerate(result["sources"]):
            preview = source["text"][:300].replace("\n", " ")

            print(f"\n{i + 1}. {source['source']} | Chunk {source['chunk_index']}")
            print(f'   Preview: "{preview}..."')