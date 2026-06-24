# RAG Behavior Testing

## Category A - Easy Questions

### Question
What is force majeure?

### Retrieved Chunks
service_agreement.pdf
Chunk 75
Chunk 76

### Answer
Force majeure refers to events beyond a party's reasonable control, such as fire, flood, earthquake, war, terrorism, riots, government actions, and other similar events.

### Assessment
Correct

### Notes
Retriever found the exact force majeure clause.
Answer was grounded in retrieved context.



### Question
What events qualify as force majeure?

### Retrieved Chunks
service_agreement.pdf
Chunk 75
Chunk 76

### Answer
Fire, flood, earthquake, acts of God, war, terrorism, riots, civil disorders, rebellions, revolutions, government actions, and other events beyond reasonable control.

### Assessment
Correct

### Notes
Answer matched the retrieved clause.
One unrelated chunk was also retrieved, showing that retrieval can contain noise.



### Question
How long does confidentiality last?

### Retrieved Chunks
nda_template.pdf
Chunk 3

### Answer
Confidentiality obligations continue until the information no longer qualifies as a trade secret or until the disclosing party releases the receiving party from the agreement.

### Assessment
Correct

### Notes
Retriever found the NDA confidentiality section.
Answer was supported by the source.



## Category C - Impossible Questions

### Question
What is the CEO's name?

### Retrieved Chunks
employee_handbook.pdf

### Answer
The context does not provide enough information.

### Assessment
Correct Refusal

### Notes
No supporting information existed in the knowledge base.
Model correctly admitted uncertainty rather than inventing an answer.



## Key Observations

- Retrieval successfully located relevant contract clauses.
- Retrieved results occasionally contained unrelated chunks.
- The LLM was still able to answer correctly when the top retrieved chunks were relevant.
- Source attribution made it easy to verify answers.
- RAG significantly reduced hallucinations for unsupported questions.

## Category B - Ambiguous Questions

### Question

Can termination occur immediately?

### Assessment

Correct

### Notes

The retriever found the termination clause. The model identified that immediate termination is allowed in insolvency scenarios but not for standard contractual breaches. Answer was supported by retrieved context.

### Question

What happens after 60 days of force majeure?

### Assessment

Partial

### Notes

The agreement only specified what happens after 30 days of force majeure. The model correctly explained the 30-day provision and stated that the document does not specify what happens after 60 days. No hallucination observed.

### Question

Who approves force majeure requests?

### Assessment

Correct Refusal

### Notes

The retrieved documents described force majeure events and notification requirements but did not mention any approval authority. The model correctly stated that the information was unavailable.

### Question

Can confidential information ever be disclosed?

### Assessment

Correct

### Notes

The model identified valid exceptions, including legal requirements and disclosures to personnel working on the agreement. The answer was supported by the retrieved confidentiality clauses.

### Question

Can an employee work another job?

### Assessment

Retrieval Failure

### Notes

The employee handbook does contain an outside employment policy, but retrieval did not find the relevant section. The model therefore stated that insufficient information was available. This demonstrates that RAG quality depends heavily on retrieval quality.
