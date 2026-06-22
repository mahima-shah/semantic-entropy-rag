from llm import ask_llm

question = input("Ask a question: ")

answer = ask_llm(question)

print("\nAnswer:\n")
print(answer)