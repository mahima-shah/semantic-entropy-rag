"""
semantic_judge.py

Uses a strict temperature-0 LLM judge to group sampled answers into
semantic meaning clusters and identify material legal disagreements.
"""

import json
from typing import Any

from llm import ask_llm


def extract_json(raw_text: str) -> dict[str, Any]:
    """
    Extract and parse a JSON object from an LLM response.
    """

    if not raw_text or not raw_text.strip():
        raise ValueError(
            "The semantic judge returned an empty response."
        )

    cleaned = (
        raw_text
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    try:
        parsed = json.loads(cleaned)

        if isinstance(parsed, dict):
            return parsed

    except json.JSONDecodeError:
        pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")

    if (
        first_brace == -1
        or last_brace == -1
        or last_brace <= first_brace
    ):
        raise ValueError(
            "The semantic judge did not return valid JSON."
        )

    extracted = cleaned[
        first_brace:last_brace + 1
    ]

    parsed = json.loads(extracted)

    if not isinstance(parsed, dict):
        raise ValueError(
            "The semantic judge response must be a JSON object."
        )

    return parsed


def format_samples(
    samples: list[dict[str, Any]]
) -> str:
    """
    Format sampled answers for the semantic judge.
    """

    sample_blocks = []

    for sample in samples:
        sample_blocks.append(
            f"Sample {sample['sample_id']}:\n"
            f"{sample['answer']}"
        )

    return "\n\n".join(sample_blocks)


def judge_samples(
    question: str,
    samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """
    Cluster answers by semantic meaning and audit contradictions.

    The judge does not determine whether an answer is legally correct.
    It only determines whether the sampled answers agree in meaning.
    """

    formatted_samples = format_samples(samples)

    prompt = f"""
You are a strict legal-answer consistency auditor.

Original question:
{question}

Generated answers:
{formatted_samples}

Your tasks:

1. Group answers that communicate the same material legal meaning.
2. Ignore differences in wording, formatting, tone, and harmless detail.
3. Create separate clusters when answers materially disagree.

A material disagreement includes:

- opposite or incompatible legal conclusions
- different governing legal rules
- different statutory provisions
- different jurisdictions
- conflicting dates
- conflicting notice periods
- conflicting monetary amounts
- conflicting parties or obligations
- conflicting exceptions
- citations to genuinely different statutes, cases, clauses, or authorities
- do not treat differences in source numbering, chunk labels, citation format,
  or wording as citation variance when they point to the same underlying
  document provision
- unsupported factual assertions

Source labels such as "Source 1" and "Source 4" may refer to the same
underlying document passage because source numbering depends on retrieval
order. Do not mark citation_variance as true merely because source numbers,
chunk labels, or citation formatting differ.

Set citation_variance to true only when the answers rely on materially
different legal authorities, statutes, cases, clauses, or document
provisions.

A more detailed answer may belong in the same cluster if it does not
contradict the other answers.

Do not determine which answer is legally correct.
Only determine whether the answers agree.

Return valid JSON only.

Use exactly this structure:

{{
  "clusters": [
    {{
      "cluster_id": 1,
      "label": "brief description of the shared meaning",
      "sample_ids": [1, 2, 3]
    }}
  ],
  "material_contradiction": false,
  "contradictions": [
    {{
      "type": "legal_conclusion",
      "sample_ids": [1, 4],
      "description": "brief explanation of the conflict"
    }}
  ],
  "citation_variance": false,
  "jurisdiction_variance": false,
  "unsupported_claim_risk": false,
  "summary": "one sentence summary of the agreement or disagreement"
}}
"""

    raw_response = ask_llm(
        prompt=prompt,
        temperature=0.0,
        max_tokens=1200
    )

    result = extract_json(raw_response)

    result.setdefault(
        "clusters",
        []
    )

    result.setdefault(
        "material_contradiction",
        False
    )

    result.setdefault(
        "contradictions",
        []
    )

    result.setdefault(
        "citation_variance",
        False
    )

    result.setdefault(
        "jurisdiction_variance",
        False
    )

    result.setdefault(
        "unsupported_claim_risk",
        False
    )

    result.setdefault(
        "summary",
        "No semantic audit summary was returned."
    )

    valid_sample_ids = {
        sample["sample_id"]
        for sample in samples
    }

    assigned_sample_ids = []

    for cluster in result["clusters"]:
        valid_ids = []

        for sample_id in cluster.get(
            "sample_ids",
            []
        ):
            try:
                sample_id = int(sample_id)
            except (TypeError, ValueError):
                continue

            if sample_id in valid_sample_ids:
                valid_ids.append(sample_id)

        cluster["sample_ids"] = valid_ids
        assigned_sample_ids.extend(valid_ids)

    missing_sample_ids = sorted(
        valid_sample_ids
        - set(assigned_sample_ids)
    )

    for sample_id in missing_sample_ids:
        result["clusters"].append({
            "cluster_id": (
                len(result["clusters"]) + 1
            ),
            "label": (
                f"Unassigned sample {sample_id}"
            ),
            "sample_ids": [sample_id]
        })

    return result