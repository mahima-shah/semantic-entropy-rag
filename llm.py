import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()

MODEL_NAME = "claude-haiku-4-5-20251001"

client = Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def ask_llm(prompt, max_tokens=700):
    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.content[0].text