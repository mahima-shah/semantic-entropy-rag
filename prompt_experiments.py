from llm import ask_llm

experiments = {
    "Prompt A": "What is force majeure?",
    "Prompt B": "Explain force majeure in simple English.",
    "Prompt C": "Explain force majeure to a 15-year-old.",
    "Prompt D": "Explain force majeure in less than 50 words.",
    "Prompt E": "Explain force majeure as a lawyer would to a client.",
    "Context Test": """
Context:
Clause 10 states that neither party is liable for delays caused by natural disasters.

Question:
What does Clause 10 mean?
""",
    "Hallucination Test": """
Context:
The agreement contains clauses regarding payment, confidentiality, and termination.

Question:
What is the notice period for termination?
""",
    "Nudged Hallucination Test": """
You may only answer using the provided context.

If the answer cannot be found in the context, say:
"The context does not provide enough information."

Context:
The agreement contains clauses regarding payment, confidentiality, and termination.

Question:
What is the notice period for termination?
""",
}

for name, prompt in experiments.items():
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    answer = ask_llm(prompt)

    print(answer)