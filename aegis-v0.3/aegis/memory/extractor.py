from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.llm_client import LLMClient
from .models import MemoryType


@dataclass
class MemoryCandidate:
    content: str
    memory_type: MemoryType
    importance: float
    confidence: float
    sensitivity: float
    tags: list[str]


EXTRACTION_PROMPT = """
You are the memory-analysis subsystem of A.E.G.I.S.

Analyze the conversation and identify information that is genuinely
useful for A.E.G.I.S. to remember across future sessions.

Only extract durable information.

Good candidates:
- Long-term user preferences
- Stable facts about the user's projects
- Important project decisions
- Long-term goals
- Important technical environment facts
- Significant completed milestones
- Explicit instructions about how A.E.G.I.S. should behave

Do NOT extract:
- Greetings
- Small talk
- Temporary thoughts
- Ordinary actions
- Every sentence containing "I"
- Information that is clearly hypothetical
- Information that is only relevant to the current turn

Be conservative.

Return ONLY valid JSON with this structure:

{
  "memories": [
    {
      "content": "short factual memory",
      "type": "semantic",
      "importance": 0.0,
      "confidence": 0.0,
      "sensitivity": 0.0,
      "tags": ["tag1"]
    }
  ]
}

Allowed types:
- working
- episodic
- semantic
- preference

Importance:
0.0 = irrelevant
1.0 = extremely important

Confidence:
0.0 = uncertain
1.0 = explicitly certain

Sensitivity:
0.0 = ordinary information
1.0 = highly sensitive personal information

Return an empty memories list if nothing deserves long-term storage.
"""


class MemoryExtractor:
    """Uses an LLM to identify possible long-term memories."""

    def __init__(self, llm_client: LLMClient):
        self._llm = llm_client

    def extract(
        self,
        messages: list[dict],
    ) -> list[MemoryCandidate]:

        conversation = self._format_messages(messages)

        response = self._llm.send(
            system=EXTRACTION_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": conversation,
                }
            ],
            tools=None,
        )

        raw = "".join(
            block.text
            for block in response.content
            if hasattr(block, "text")
        )

        return self._parse(raw)

    @staticmethod
    def _format_messages(messages: list[dict]) -> str:
        lines = []

        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")

            if isinstance(content, str):
                lines.append(
                    f"{role.upper()}: {content}"
                )

        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> list[MemoryCandidate]:
        raw = raw.strip()

        if not raw:
            return []

        # Handle accidental markdown fences.
        raw = re.sub(
            r"^```(?:json)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )

        raw = re.sub(
            r"\s*```$",
            "",
            raw,
        )

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, dict):
            return []

        memories = data.get("memories", [])

        if not isinstance(memories, list):
            return []

        candidates = []

        for item in memories:
            if not isinstance(item, dict):
                continue

            content = str(
                item.get("content", "")
            ).strip()

            if not content:
                continue

            try:
                memory_type = MemoryType(
                    item.get(
                        "type",
                        "semantic",
                    )
                )
            except ValueError:
                continue

            try:
                importance = float(
                    item.get(
                        "importance",
                        0.0,
                    )
                )

                confidence = float(
                    item.get(
                        "confidence",
                        0.0,
                    )
                )

                sensitivity = float(
                    item.get(
                        "sensitivity",
                        0.0,
                    )
                )

            except (TypeError, ValueError):
                continue

            tags = item.get("tags", [])

            if not isinstance(tags, list):
                tags = []

            candidates.append(
                MemoryCandidate(
                    content=content,
                    memory_type=memory_type,
                    importance=max(
                        0.0,
                        min(1.0, importance),
                    ),
                    confidence=max(
                        0.0,
                        min(1.0, confidence),
                    ),
                    sensitivity=max(
                        0.0,
                        min(1.0, sensitivity),
                    ),
                    tags=[
                        str(tag)
                        for tag in tags
                    ],
                )
            )

        return candidates