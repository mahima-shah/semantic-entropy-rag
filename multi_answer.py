"""
multi_answer.py

Supports two different reliability experiments:

1. Fixed-temperature sampling:
   Repeats the same question at one temperature to estimate semantic
   uncertainty.

2. Temperature sweep:
   Generates answers at different temperatures to measure how sensitive
   the answer is to generation settings.
"""

from typing import Any

from llm import ask_llm
from rag import build_prompt


DEFAULT_SAMPLE_COUNT = 5
DEFAULT_SAMPLE_TEMPERATURE = 0.7
DEFAULT_TEMPERATURE_SWEEP = [
    0.0,
    0.3,
    0.6,
    0.9
]


def generate_fixed_temperature_answers(
    question: str,
    chunks: list[dict[str, Any]],
    sample_count: int = DEFAULT_SAMPLE_COUNT,
    temperature: float = DEFAULT_SAMPLE_TEMPERATURE
) -> list[dict[str, Any]]:
    """
    Generate multiple answers using the same temperature.

    This is the sampling method used for the semantic uncertainty test.
    """

    if sample_count < 2:
        raise ValueError(
            "sample_count must be at least 2."
        )

    prompt = build_prompt(
        question,
        chunks
    )

    samples = []

    for index in range(sample_count):
        answer = ask_llm(
            prompt=prompt,
            temperature=temperature,
            max_tokens=900
        )

        samples.append({
            "sample_id": index + 1,
            "temperature": temperature,
            "answer": answer
        })

    return samples


def generate_temperature_sweep_answers(
    question: str,
    chunks: list[dict[str, Any]],
    temperatures: list[float] | None = None
) -> list[dict[str, Any]]:
    """
    Generate one answer at each temperature.

    This measures sensitivity to the temperature setting. It is not used
    for the semantic entropy calculation.
    """

    if temperatures is None:
        temperatures = (
            DEFAULT_TEMPERATURE_SWEEP
        )

    prompt = build_prompt(
        question,
        chunks
    )

    samples = []

    for index, temperature in enumerate(
        temperatures
    ):
        answer = ask_llm(
            prompt=prompt,
            temperature=temperature,
            max_tokens=900
        )

        samples.append({
            "sample_id": index + 1,
            "temperature": temperature,
            "answer": answer
        })

    return samples