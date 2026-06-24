from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DATA_DIR = Path("data")
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "legal_docs"

print("Loading embedding model... this may take a few minutes the first time")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding model loaded.")

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def build_database():
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    pdf_files = list(DATA_DIR.glob("*.pdf"))

    all_texts = []
    all_ids = []
    all_metadatas = []

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

    embeddings = embedding_model.encode(all_texts).tolist()

    collection.add(
        documents=all_texts,
        embeddings=embeddings,
        metadatas=all_metadatas,
        ids=all_ids
    )

    print(f"\nDatabase built successfully.")
    print(f"Total chunks stored: {len(all_texts)}")


def search(query, top_k=3):
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


if __name__ == "__main__":
    build_database()

    while True:
        query = input("\nAsk a retrieval question or type 'exit': ")

        if query.lower() == "exit":
            break

        search(query)