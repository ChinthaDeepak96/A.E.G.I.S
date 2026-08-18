"""
Core configuration for A.E.G.I.S. v0.1 (Core Brain).

Loads settings from environment variables / a .env file. Kept dumb
on purpose -- v0.1 has exactly one model provider and no routing
logic yet. The Model Router (architecture doc section 51) gets built
once there's more than one model worth routing between.
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
    anthropic_api_key: str
    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    history_limit: int = 40  # messages kept in working memory before truncation


def load_settings() -> Settings:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env "
            "and add your key, or export it in your shell."
        )
    model = os.environ.get("AEGIS_MODEL", "claude-sonnet-4-6")
    return Settings(anthropic_api_key=api_key, model=model)
