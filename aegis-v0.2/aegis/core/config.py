"""
Core configuration for A.E.G.I.S. Loads settings from environment
variables / a .env file.

AEGIS_PROVIDER selects which LLMClient gets built (core/llm_client.py):
  - "anthropic" (default): paid API, requires ANTHROPIC_API_KEY.
  - "local": free, runs against a local Ollama server, no key needed.

This is the first piece of the future Model Router (architecture doc
section 51) -- for now it's a single switch, not real routing logic.
"""

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    # python-dotenv is optional; env vars can be set directly instead.
    pass


@dataclass(frozen=True)
class Settings:
    provider: str = "anthropic"
    anthropic_api_key: str = ""
    model: str = "claude-sonnet-4-6"
    ollama_model: str = "llama3.1"
    ollama_host: str = "http://localhost:11434"
    max_tokens: int = 1024
    history_limit: int = 40  # messages kept in working memory before truncation


def load_settings() -> Settings:
    provider = os.environ.get("AEGIS_PROVIDER", "anthropic").strip().lower()
    max_tokens = int(os.environ.get("AEGIS_MAX_TOKENS", "1024"))
    history_limit = int(os.environ.get("AEGIS_HISTORY_LIMIT", "40"))

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or set AEGIS_PROVIDER=local to run MAX for free on a "
                "local Ollama model instead."
            )
        model = os.environ.get("AEGIS_MODEL", "claude-sonnet-4-6")
        return Settings(
            provider=provider,
            anthropic_api_key=api_key,
            model=model,
            max_tokens=max_tokens,
            history_limit=history_limit,
        )

    if provider in ("local", "ollama"):
        ollama_model = os.environ.get("AEGIS_OLLAMA_MODEL", "llama3.1")
        ollama_host = os.environ.get("AEGIS_OLLAMA_HOST", "http://localhost:11434")
        return Settings(
            provider=provider,
            ollama_model=ollama_model,
            ollama_host=ollama_host,
            max_tokens=max_tokens,
            history_limit=history_limit,
        )

    raise RuntimeError(f"Unknown AEGIS_PROVIDER '{provider}'. Use 'anthropic' or 'local'.")
