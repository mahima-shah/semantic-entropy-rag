"""
llm.py

Utility functions for interacting with the Anthropic Claude API.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv


load_dotenv()

api_key = os.getenv(
    "ANTHROPIC_API_KEY"
)

model_name = os.getenv(
    "ANTHROPIC_MODEL",
    "claude-haiku-4-5-20251001"
)

if not api_key:
    raise ValueError(
        "ANTHROPIC_API_KEY is missing. "
        "Add it to your .env file."
    )

client = Anthropic(
    api_key=api_key
)


def ask_llm(
    prompt: str,
    max_tokens: int = 900,
    temperature: float = 0.0
) -> str:
    """
    Send a prompt to Claude and return generated text.
    """

    response = client.messages.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    text_parts = []

    for block in response.content:
        if getattr(
            block,
            "type",
            None
        ) == "text":
            text_parts.append(
                block.text
            )

    if not text_parts:
        raise RuntimeError(
            "Claude returned no text response."
        )

    return "\n".join(
        text_parts
    ).strip()