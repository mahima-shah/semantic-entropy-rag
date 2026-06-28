from rag import retrieve_chunks
from llm import ask_llm
from entropy import compare_all_answers, confidence_label


def format_chunks_for_context(chunks):
    context = ""

    for i, chunk in enumerate(chunks):
        context += f"\n[Source {i+1}: {chunk['source']} | Chunk {chunk['chunk_index']}]\n"
        context += chunk["text"]
        context += "\n"

    return context


def build_nudged_prompt(question, chunks):
    context = format_chunks_for_context(chunks)

    return f"""
You are answering questions using only the retrieved context.

Rules:
1. Use only the context below.
2. Do not use outside knowledge.
3. Do not guess missing facts.
4. Do not infer dates, names, notice periods, duties, approval processes, or legal consequences unless explicitly stated.
5. If the context does not contain the answer, say:
   "The provided documents do not contain enough information to answer this."
6. If the context is ambiguous, say what is clear and what is unclear.
7. Cite the source numbers used.

Context:
{context}

Question:
{question}

Answer:
"""


def generate_nudged_answer(question, chunks):
    prompt = build_nudged_prompt(question, chunks)
    return ask_llm(prompt)


def confidence_from_answers(answers):
    comparisons = compare_all_answers(answers)
    confidence = confidence_label(comparisons)

    return {
        "comparisons": comparisons,
        "confidence": confidence
    }


if __name__ == "__main__":
    question = input("\nAsk a question to nudge: ")

    chunks = retrieve_chunks(question)

    original_answers = []

    print("\nGenerating original answers...")
    for i in range(3):
        from rag import build_prompt
        original_prompt = build_prompt(question, chunks)
        original_answer = ask_llm(original_prompt)
        original_answers.append(original_answer)

    original_confidence_data = confidence_from_answers(original_answers)

    print("\nOriginal Answers:")
    for i, answer in enumerate(original_answers):
        print("\n" + "=" * 80)
        print(f"Original Answer {i+1}")
        print("=" * 80)
        print(answer)

    print("\nOriginal Comparisons:")
    for item in original_confidence_data["comparisons"]:
        print(f"{item['pair']}: {item['result']}")

    print("\nOriginal Confidence:")
    print(original_confidence_data["confidence"])

    nudged_answer = None

    if original_confidence_data["confidence"] == "Low":
        print("\nLow confidence detected. Applying nudged prompt...")
        nudged_answer = generate_nudged_answer(question, chunks)
    else:
        print("\nConfidence is not Low. Nudging not required.")
        nudged_answer = generate_nudged_answer(question, chunks)

    print("\nNudged Answer:")
    print(nudged_answer)

    print("\nSources:")
    for i, chunk in enumerate(chunks):
        preview = chunk["text"][:300].replace("\n", " ")
        print(f"\n{i+1}. {chunk['source']} | Chunk {chunk['chunk_index']}")
        print(f'   Preview: "{preview}..."')