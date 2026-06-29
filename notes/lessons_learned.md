# Lessons Learned

## Technical

### What did I learn about RAG?

I learned that Retrieval-Augmented Generation (RAG) improves factual grounding by retrieving relevant document chunks before generating an answer. However, retrieval alone does not guarantee correctness, as the language model can still misinterpret the retrieved context or make unsupported assumptions.

---

### What surprised me about retrieval?

Even when the correct information existed in the vector database, the quality of the retrieved chunks had a significant impact on the final answer. Small differences in retrieval quality often resulted in noticeably different responses.

---

### Why isn't RAG enough by itself?

RAG reduces hallucinations but does not eliminate them. The language model may still produce confident answers that are only partially supported by the retrieved context or answer questions whose information is missing.

---

### What did semantic entropy add?

Generating multiple answers and comparing their meanings provided a simple way to estimate uncertainty. Instead of treating every answer equally, the system could identify cases where the model produced competing interpretations and lower its confidence accordingly.

---

### When did nudging help?

Prompt-based nudging was most useful for low-confidence responses. The stricter prompt encouraged the model to rely only on retrieved evidence, explicitly acknowledge missing information, and avoid unsupported assumptions.

---

## Engineering

### Design Decisions

The project uses a lightweight semantic entropy-inspired approach based on multiple answer generation and pairwise LLM comparison. This was chosen because it is significantly simpler to implement than the original research algorithm while still demonstrating the underlying concept.

---

### What would I do differently?

With more time, I would implement hybrid retrieval, reranking, embedding-based semantic clustering, and confidence calibration using a larger evaluation dataset.

---

### Biggest Challenges

The main implementation challenges involved integrating the retrieval pipeline, handling package and environment issues, designing meaningful confidence heuristics, and ensuring that all components worked together through a Streamlit interface.

---

## Project Reflection

### Current Limitations

This project is a prototype and does not implement the original semantic entropy algorithm. Confidence scores are heuristic rather than mathematically calibrated, and evaluation was performed on a relatively small document collection.

---

### Future Improvements

1. Hybrid keyword and vector retrieval.
2. Embedding-based semantic clustering instead of pairwise LLM comparison.
3. Larger evaluation datasets with automated confidence calibration.
