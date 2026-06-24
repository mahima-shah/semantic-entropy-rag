# Confidence Logic

## Goal

Convert answer agreement into an interpretable confidence score.

## Method

Three answers are generated from the same retrieved context.

The answers are compared pairwise:

- A1 vs A2
- A1 vs A3
- A2 vs A3

Each comparison returns:

YES = same meaning

NO = different meaning

## Confidence Rules

### High

All comparisons agree.

Example:

A1 vs A2 = YES
A1 vs A3 = YES
A2 vs A3 = YES

Interpretation:
The model consistently produced the same meaning.

### Medium

One disagreement.

Example:

A1 vs A2 = YES
A1 vs A3 = YES
A2 vs A3 = NO

Interpretation:
Most answers agree, but some uncertainty exists.

### Low

Two or more disagreements.

Example:

A1 vs A2 = YES
A1 vs A3 = NO
A2 vs A3 = NO

Interpretation:
Multiple competing interpretations exist.
The answer should be treated with caution.

## Key Learning

Confidence is not based on whether the answer is correct.

Confidence is based on whether multiple independently generated answers communicate the same meaning.

This is the core idea behind semantic entropy.