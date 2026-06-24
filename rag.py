print("Importing sentence-transformers...")

from sentence_transformers import SentenceTransformer
import chromadb
from llm import ask_llm

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "legal_docs"

print("Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")


def retrieve_chunks(query, top_k=3):
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


def build_prompt(question, chunks):
    context = ""

    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']} | Chunk {chunk['chunk_index']}]\n"
        context += chunk["text"]
        context += "\n"

    prompt = f"""
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

    return prompt


def answer_question(question):
    chunks = retrieve_chunks(question)
    prompt = build_prompt(question, chunks)
    answer = ask_llm(prompt)

    return {
        "question": question,
        "answer": answer,
        "sources": chunks
    }


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

            print(f"\n{i+1}. {source['source']} | Chunk {source['chunk_index']}")
            print(f'   Preview: "{preview}..."')