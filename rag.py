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

def retrieve_chunks(query: str, top_k: int = 3) -> list[dict[str, Any]]:
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


# -----------------------------
# Prompt Construction
# -----------------------------

def build_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """
    Build the RAG prompt using retrieved context.

    The prompt explicitly instructs the model to answer only from
    the retrieved chunks and to avoid unsupported guesses.

    Args:
        question:
            User question.

        chunks:
            Retrieved document chunks.

    Returns:
        Prompt string to send to the LLM.
    """

    context_parts = []

    for i, chunk in enumerate(chunks):
        source_header = (
            f"[Source {i + 1}: "
            f"{chunk['source']} | Chunk {chunk['chunk_index']}]"
        )

        context_parts.append(f"{source_header}\n{chunk['text']}")

    context = "\n\n".join(context_parts)

    return f"""
You are answering questions using only the provided context.

Rules:
- Use only the context below.
- If the answer is not in the context, say: "The context does not provide enough information."
- Do not guess.
- Do not use outside knowledge.
- Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""


# -----------------------------
# End-to-End RAG
# -----------------------------

def answer_question(question: str) -> dict[str, Any]:
    """
    Run the full RAG pipeline for a single question.

    Args:
        question:
            User question.

    Returns:
        Dictionary containing the question, answer, and retrieved sources.
    """

    chunks = retrieve_chunks(question)
    prompt = build_prompt(question, chunks)
    answer = ask_llm(prompt)

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