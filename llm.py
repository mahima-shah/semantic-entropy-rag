"""
llm.py

Utility functions for interacting with the Anthropic Claude API.

This module loads the API key from the environment and provides a
simple wrapper function for sending prompts to the language model.
"""

import os

from anthropic import Anthropic
from dotenv import load_dotenv


# Load environment variables from the .env file
load_dotenv()


# Initialize the Anthropic client
client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)


def ask_llm(prompt: str, max_tokens: int = 700) -> str:
    """
    Sends a prompt to the Anthropic Claude model and returns the response.

    Args:
        prompt (str):
            The prompt to send to the language model.

        max_tokens (int):
            Maximum number of tokens to generate.
            Default is 700.

    Returns:
        str:
            The generated response from Claude.
    """

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text