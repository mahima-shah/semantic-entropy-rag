# RAG Reliability Checker

A lightweight Retrieval-Augmented Generation (RAG) system enhanced with a semantic entropy-inspired confidence layer and prompt-based nudging to improve answer reliability and transparency.

---

# Overview

Retrieval-Augmented Generation (RAG) systems reduce hallucinations by retrieving relevant documents before generating an answer. However, retrieval alone does not eliminate incorrect or overconfident responses. Even when relevant information is retrieved, the language model may interpret it differently, make unsupported assumptions, or answer questions that are only partially covered by the available context.

This project explores whether answer reliability can be improved by estimating confidence through a semantic entropy-inspired approach. Instead of generating a single response, the system generates multiple answers using the same retrieved context, compares their meanings, assigns a confidence score, and applies prompt-based nudging whenever confidence is low.

The objective is not to eliminate hallucinations completely, but to make uncertainty more transparent to the user.

---

# Project Goal

Build an end-to-end RAG system capable of:

* Retrieving relevant document chunks
* Answering questions using retrieved context
* Estimating confidence through multiple answer agreement
* Applying prompt-based nudging when confidence is low
* Displaying sources for answer verification

---

# System Architecture

```text
User Question
       │
       ▼
Sentence Transformer Embedding
       │
       ▼
ChromaDB Vector Search
       │
       ▼
Top Retrieved Chunks
       │
       ▼
LLM Answer Generation
       │
       ▼
Generate 3 Independent Answers
       │
       ▼
Pairwise Meaning Comparison
       │
       ▼
Confidence Score
 (High / Medium / Low)
       │
       ▼
Low Confidence?
       │
   ┌───┴────┐
   │        │
  No       Yes
   │        │
   │   Prompt-Based Nudging
   │        │
   └───┬────┘
       ▼
Final Answer + Sources
```

---

# Features

* Retrieval-Augmented Generation (RAG)
* PDF document ingestion
* Automatic document chunking
* Sentence Transformer embeddings
* ChromaDB vector database
* Semantic search retrieval
* Semantic entropy-inspired confidence estimation
* Multiple answer generation
* Meaning comparison using an LLM
* Confidence scoring (High, Medium, Low)
* Prompt-based nudging
* Source attribution
* Interactive Streamlit interface
* Evaluation dataset and automated result logging

---

# Tech Stack

* Python
* Streamlit
* ChromaDB
* Sentence Transformers
* Transformers
* PyTorch
* Anthropic Claude API
* Pandas
* PDF Processing (PyPDF)

---

# Project Structure

```text
semantic-entropy-rag/

├── app.py
├── rag.py
├── retrieval.py
├── entropy.py
├── nudging.py
├── compare_answers.py
├── multi_answer.py
├── llm.py
├── tests/
├── results/
├── notes/
├── data/
├── screenshots/
├── docs/
└── README.md
```

---

# Evaluation

The system was evaluated using 27 questions divided into three categories:

* Easy
* Ambiguous
* Impossible

Each evaluation measured:

* Answer quality
* Confidence score
* Source attribution
* Nudged response

Results showed that the confidence layer generally reflected answer uncertainty and that prompt-based nudging produced more cautious responses when confidence was low.

---

# Future Improvements

* Hybrid retrieval (keyword + vector search)
* Embedding-based semantic clustering
* Cross-encoder reranking
* Larger evaluation datasets
* Automatic confidence calibration
* Multi-document retrieval
* True semantic entropy implementation

---

# Acknowledgements

This project was developed as part of an AI internship to explore techniques for improving the trustworthiness of Retrieval-Augmented Generation systems.

## What This Project Is Not

This project is a lightweight prototype inspired by research on semantic entropy for uncertainty estimation in language models.

It does **not** implement the original semantic entropy algorithm described in the research literature. Instead, it uses a practical approximation based on:

- Multiple answer generation
- Pairwise semantic comparison using an LLM
- Heuristic confidence scoring
- Prompt-based nudging for low-confidence responses

The evaluation is performed on a relatively small document collection and is intended to demonstrate the concept rather than provide production-grade confidence calibration.