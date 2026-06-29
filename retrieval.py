"""
retrieval.py

Builds and queries the ChromaDB vector database.

This file handles:
1. Reading PDF documents from the data/ folder
2. Splitting documents into overlapping text chunks
3. Creating embeddings for each chunk
4. Storing chunks and metadata in ChromaDB
5. Running simple retrieval tests from the terminal
"""

from pathlib import Path
from typing import Any

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# -----------------------------
# Configuration
# -----------------------------

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "legal_docs"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


# Load embedding model once so it can be reused for both indexing and querying.
print("Loading embedding model... this may take a few minutes the first time.")
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")


# -----------------------------
# PDF Reading
# -----------------------------

def read_pdf(file_path: Path) -> str:
    """
    Extract text from a PDF file.

    Args:
        file_path:
            Path to the PDF file.

    Returns:
        Extracted text from all readable pages.
    """

    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


# -----------------------------
# Chunking
# -----------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into overlapping chunks.

    Overlap helps preserve context across chunk boundaries so that clauses
    are less likely to be cut off completely.

    Args:
        text:
            Full document text.

        chunk_size:
            Maximum number of characters per chunk.

        overlap:
            Number of characters repeated between adjacent chunks.

    Returns:
        List of text chunks.
    """

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# -----------------------------
# Database Construction
# -----------------------------

def build_database() -> None:
    """
    Build a fresh ChromaDB collection from all PDFs in the data folder.

    Existing collections with the same name are deleted so each run creates
    a clean index.
    """

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        # Collection may not exist yet, which is fine.
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data folder.")
        return

    all_texts: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict[str, Any]] = []

    chunk_id = 0

    for pdf_path in pdf_files:
        print(f"Reading: {pdf_path.name}")

        text = read_pdf(pdf_path)
        chunks = chunk_text(text)

        for index, chunk in enumerate(chunks):
            all_texts.append(chunk)
            all_ids.append(f"chunk_{chunk_id}")
            all_metadatas.append({
                "source": pdf_path.name,
                "chunk_index": index
            })

            chunk_id += 1

    print("Generating embeddings...")
    embeddings = embedding_model.encode(all_texts).tolist()

    collection.add(
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print("\nDatabase built successfully.")
    print(f"Total chunks stored: {len(all_texts)}")


# -----------------------------
# Retrieval
# -----------------------------

def search(query: str, top_k: int = 3) -> None:
    """
    Search the vector database and print the top retrieved chunks.

    This function is mainly used for terminal testing before connecting
    retrieval to the LLM.

    Args:
        query:
            User question or search query.

        top_k:
            Number of chunks to retrieve.
    """

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    query_embedding = embedding_model.encode([query]).tolist()[0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    print("\nQuestion:")
    print(query)

    print("\nRetrieved Chunks:")

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        print("\n" + "-" * 80)
        print(f"Result {i + 1}")
        print(f"Source: {metadata['source']}")
        print(f"Chunk index: {metadata['chunk_index']}")
        print("-" * 80)
        print(document[:1000])


# -----------------------------
# Command-Line Entry Point
# -----------------------------

if __name__ == "__main__":
    build_database()

    while True:
        query = input("\nAsk a retrieval question or type 'exit': ")

        if query.lower() == "exit":
            break

        search(query)