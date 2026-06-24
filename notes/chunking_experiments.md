# Chunking Experiments

## Documents Tested

- employee_handbook.pdf
- openai_privacy.pdf
- service_agreement.pdf
- nda_template.pdf
- openai_terms.pdf

## Observations

### Chunk size 300
Produced many chunks. This is too small for legal and policy documents because clauses and sections can get split awkwardly.

### Chunk size 500
More readable than 300, but longer clauses may still be split.

### Chunk size 1000
Preserves more context and keeps clauses together better. This seems best for the first version of the project.

## Decision

Use:

chunk_size = 1000  
overlap = 150

Reason: legal and policy documents need surrounding context. Smaller chunks may retrieve only half a clause.