from __future__ import annotations
import os

try:
    from anthropic import Anthropic
except ImportError:  # allows the analytics/tests to run without the package installed
    Anthropic = None

DEFAULT_MODEL = "claude-sonnet-4-6"


def get_client():
    if Anthropic is None:
        raise ImportError("The 'anthropic' package is not installed. Run: pip install anthropic")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set. Add it to your .env file.")

    return Anthropic(api_key=api_key)


def ask_claude(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL, max_tokens: int = 2000) -> str:
    client = get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text
