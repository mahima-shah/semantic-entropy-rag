from rag import retrieve_chunks, build_prompt
from llm import ask_llm


def generate_multiple_answers(question, num_answers=3):
    chunks = retrieve_chunks(question)

    answers = []

    for i in range(num_answers):
        prompt = build_prompt(question, chunks)

        prompt += f"""

Generate answer version {i + 1}.
Keep the answer grounded in the provided context.
"""

        answer = ask_llm(prompt)
        answers.append(answer)

    return {
        "question": question,
        "chunks": chunks,
        "answers": answers
    }


if __name__ == "__main__":
    while True:
        question = input("\nAsk a question for multiple answers or type 'exit': ")

        if question.lower() == "exit":
            break

        result = generate_multiple_answers(question)

        print("\nQuestion:")
        print(result["question"])

        print("\nRetrieved Sources:")
        for i, chunk in enumerate(result["chunks"]):
            print(f"{i+1}. {chunk['source']} | Chunk {chunk['chunk_index']}")

        print("\nGenerated Answers:")
        for i, answer in enumerate(result["answers"]):
            print("\n" + "=" * 80)
            print(f"Answer {i+1}")
            print("=" * 80)
            print(answer)