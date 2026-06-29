# Baseline Comparison

| Scenario              | Plain RAG                                  | RAG + Confidence + Nudging                                                   |
| --------------------- | ------------------------------------------ | ---------------------------------------------------------------------------- |
| Easy factual question | Correct answer                             | Correct answer with confidence score and sources                             |
| Ambiguous question    | May answer confidently despite uncertainty | Confidence lowered when generated answers disagree                           |
| Impossible question   | May refuse or hallucinate                  | Refuses unsupported questions with high confidence                           |
| Retrieval failure     | May produce unsupported answer             | Low confidence triggers prompt-based nudging                                 |
| User transparency     | Answer only                                | Answer, confidence score, retrieved sources, and nudged response when needed |

## Key Takeaway

The semantic entropy-inspired confidence layer does not improve retrieval or change the underlying language model. Instead, it improves transparency by estimating answer reliability and encouraging safer responses when uncertainty is detected. This allows users to better judge when an answer should be trusted and when additional verification may be required.
