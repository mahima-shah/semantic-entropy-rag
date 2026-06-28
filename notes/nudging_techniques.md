# Nudging Techniques

## Goal

Deep nudging is used when the system detects low confidence or uncertainty. The goal is to make the model answer more safely by restricting it to the retrieved evidence and preventing unsupported assumptions.

## Nudges

### 1. Context-Only Nudge

The model is instructed to answer only using the retrieved context.

Purpose:
Prevents the model from using outside knowledge or general assumptions.

Prompt rule:
Use only the retrieved context below.

---

### 2. Insufficient-Information Nudge

The model is instructed to say when the context does not contain the answer.

Purpose:
Prevents hallucination when retrieval does not provide enough evidence.

Prompt rule:
If the context does not contain the answer, say:
"The provided documents do not contain enough information to answer this."

---

### 3. Evidence-First Nudge

The model must identify or quote the supporting evidence before answering.

Purpose:
Forces the answer to be grounded in an actual retrieved chunk.

Prompt rule:
Before answering, identify the specific evidence from the context that supports the answer.

---

### 4. No-Assumption Nudge

The model is instructed not to infer missing facts.

Purpose:
Prevents the model from filling blanks with likely but unsupported information.

Prompt rule:
Do not infer dates, names, notice periods, duties, or approval processes unless explicitly stated in the context.

---

### 5. Conflict-Aware Nudge

The model must mention when retrieved chunks conflict or are incomplete.

Purpose:
Prevents the model from choosing one interpretation when the retrieved evidence is unclear.

Prompt rule:
If the retrieved context is conflicting or ambiguous, explain what is clear and what is unclear.