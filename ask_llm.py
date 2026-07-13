def ask_llm(
    prompt: str,
    max_tokens: int = 700,
    temperature: float = 0.0
) -> str:
    """
    Send a prompt to Claude and return the generated text.

    Args:
        prompt:
            Prompt sent to the language model.

        max_tokens:
            Maximum number of output tokens.

        temperature:
            Controls sampling variation.
            Lower values produce more stable outputs.
            Higher values allow more variation.

    Returns:
        Generated response text.
    """

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text