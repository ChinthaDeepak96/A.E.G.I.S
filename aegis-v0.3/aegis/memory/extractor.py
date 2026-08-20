"""
A.E.G.I.S. Automatic Memory Extraction.

The extractor asks the configured LLM to identify potentially useful
persistent memories from conversation history.

Important architectural rule:

    Conversation
        ↓
    MemoryExtractor
        ↓
    MemoryCandidate
        ↓
    MemoryManager
        ↓
    Policy / Conflict Detection / Resolution
        ↓
    SQLite

This module does NOT:
    - write to SQLite
    - modify existing memories
    - resolve conflicts
    - bypass MemoryPolicy

The LLM only proposes memory candidates. The MemoryManager remains
responsible for deciding whether those candidates are actually stored.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .models import MemoryType


@dataclass(frozen=True)
class MemoryCandidate:
    """
    A proposed memory extracted from conversation history.

    This is intentionally not a Memory object. It has no database ID
    or lifecycle state because it has not been accepted by the memory
    governance pipeline yet.
    """

    content: str

    memory_type: MemoryType

    importance: float

    confidence: float

    sensitivity: float = 0.0

    source: str = "automatic_extraction"

    tags: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)


EXTRACTION_SYSTEM_PROMPT = """
You are the A.E.G.I.S. memory extraction subsystem.

Your task is to identify information from the conversation that is
useful to remember across future conversations.

Only extract information that is reasonably persistent and useful.

Good candidates include:
- user preferences
- stable facts about the user's projects
- important technical decisions
- persistent configuration choices
- meaningful project information
- facts that will improve future assistance

Do NOT extract:
- greetings
- ordinary questions
- temporary requests
- commands
- casual conversation
- obvious one-time instructions
- information that is not useful later
- secrets, passwords, API keys, authentication tokens, or credentials

Return ONLY valid JSON.

Required format:

{
  "memories": [
    {
      "content": "memory statement",
      "type": "semantic",
      "importance": 0.0,
      "confidence": 0.0,
      "sensitivity": 0.0,
      "tags": []
    }
  ]
}

Allowed types:
- working
- episodic
- semantic
- preference

All scores must be between 0.0 and 1.0.

If there is nothing worth remembering, return:

{
  "memories": []
}
""".strip()


class MemoryExtractor:
    """
    LLM-backed automatic memory extractor.

    The extractor is deliberately limited to producing candidates.
    It never persists them.
    """

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def extract(
        self,
        messages: list[dict],
    ) -> list[MemoryCandidate]:
        """
        Extract memory candidates from conversation messages.

        Invalid or malformed LLM output is treated as an empty
        extraction rather than crashing the conversation pipeline.
        """

        if not messages:
            return []

        conversation = self._format_messages(messages)

        if not conversation.strip():
            return []

        try:
            response = self.llm_client.send(
                system=EXTRACTION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": conversation,
                    }
                ],
                tools=None,
            )
        except Exception:
            # Memory extraction must never break the main assistant.
            return []

        text = self._extract_text(response)

        if not text:
            return []

        payload = self._parse_json(text)

        if not isinstance(payload, dict):
            return []

        raw_memories = payload.get("memories")

        if not isinstance(raw_memories, list):
            return []

        candidates: list[MemoryCandidate] = []

        for raw_memory in raw_memories:
            candidate = self._build_candidate(
                raw_memory
            )

            if candidate is not None:
                candidates.append(candidate)

        return candidates

    # =========================================================
    # LLM response handling
    # =========================================================

    @staticmethod
    def _extract_text(response) -> str:
        """
        Extract text blocks from the provider-neutral LLMResponse.
        """

        content = getattr(
            response,
            "content",
            None,
        )

        if not isinstance(content, list):
            return ""

        parts: list[str] = []

        for block in content:
            text = getattr(
                block,
                "text",
                None,
            )

            if isinstance(text, str):
                parts.append(text)

        return "\n".join(parts).strip()

    @staticmethod
    def _parse_json(text: str) -> Any:
        """
        Parse JSON safely.

        Also handles models that unnecessarily wrap JSON in a
        markdown code fence.
        """

        text = text.strip()

        if not text:
            return None

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        fenced = re.search(
            r"```(?:json)?\s*(.*?)\s*```",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )

        if fenced:
            try:
                return json.loads(
                    fenced.group(1)
                )
            except json.JSONDecodeError:
                return None

        return None

    # =========================================================
    # Candidate construction
    # =========================================================

    def _build_candidate(
        self,
        raw: Any,
    ) -> MemoryCandidate | None:
        """
        Validate and normalize one raw LLM memory object.
        """

        if not isinstance(raw, dict):
            return None

        content = raw.get("content")

        if not isinstance(content, str):
            return None

        content = " ".join(
            content.strip().split()
        )

        if not content:
            return None

        memory_type = self._parse_memory_type(
            raw.get("type")
        )

        if memory_type is None:
            return None

        importance = self._clamp(
            raw.get("importance", 0.5)
        )

        confidence = self._clamp(
            raw.get("confidence", 0.5)
        )

        sensitivity = self._clamp(
            raw.get("sensitivity", 0.0)
        )

        tags = self._normalize_tags(
            raw.get("tags")
        )

        # Automatically extracted memories are always marked with
        # their extraction origin.
        metadata = {
            "extraction_method": "llm",
        }

        return MemoryCandidate(
            content=content,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            sensitivity=sensitivity,
            source="automatic_extraction",
            tags=tags,
            metadata=metadata,
        )

    @staticmethod
    def _parse_memory_type(
        value: Any,
    ) -> MemoryType | None:
        if isinstance(value, MemoryType):
            return value

        if not isinstance(value, str):
            return None

        try:
            return MemoryType(
                value.strip().lower()
            )
        except ValueError:
            return None

    @staticmethod
    def _clamp(
        value: Any,
        default: float = 0.5,
    ) -> float:
        try:
            number = float(value)
        except (
            TypeError,
            ValueError,
        ):
            number = default

        return max(
            0.0,
            min(
                1.0,
                number,
            ),
        )

    @staticmethod
    def _normalize_tags(
        tags: Any,
    ) -> list[str]:
        if not isinstance(tags, list):
            return []

        normalized: list[str] = []

        for tag in tags:
            if not isinstance(tag, str):
                continue

            tag = tag.strip().lower()

            if tag and tag not in normalized:
                normalized.append(tag)

        return normalized

    # =========================================================
    # Conversation formatting
    # =========================================================

    @staticmethod
    def _format_messages(
        messages: list[dict],
    ) -> str:
        """
        Convert the existing A.E.G.I.S. conversation format into
        plain text for the extraction prompt.
        """

        lines: list[str] = []

        for message in messages:
            if not isinstance(
                message,
                dict,
            ):
                continue

            role = str(
                message.get(
                    "role",
                    "",
                )
            ).strip()

            content = message.get(
                "content",
                "",
            )

            if isinstance(
                content,
                str,
            ):
                content_text = content.strip()

            elif isinstance(
                content,
                list,
            ):
                content_parts: list[str] = []

                for block in content:
                    if not isinstance(
                        block,
                        dict,
                    ):
                        continue

                    if block.get(
                        "type"
                    ) == "text":
                        text = block.get(
                            "text",
                            "",
                        )

                        if isinstance(
                            text,
                            str,
                        ):
                            content_parts.append(
                                text
                            )

                content_text = "\n".join(
                    content_parts
                ).strip()

            else:
                continue

            if not content_text:
                continue

            if role == "user":
                label = "User"

            elif role == "assistant":
                label = "AEGIS"

            else:
                label = role.capitalize()

            lines.append(
                f"{label}: {content_text}"
            )

        return "\n".join(lines)