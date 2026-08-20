from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
import json
import uuid


class MemoryType(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PREFERENCE = "preference"


@dataclass
class Memory:
    content: str
    memory_type: MemoryType = MemoryType.EPISODIC
    importance: float = 0.5
    confidence: float = 1.0
    sensitivity: float = 0.0
    source: str = "conversation"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        self.importance = max(0.0, min(1.0, float(self.importance)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.sensitivity = max(0.0, min(1.0, float(self.sensitivity)))

    def to_record(self) -> tuple:
        return (
            self.id, self.memory_type.value, self.content, self.importance,
            self.confidence, self.sensitivity, self.source,
            json.dumps(self.tags, ensure_ascii=False),
            json.dumps(self.metadata, ensure_ascii=False),
            self.created_at, self.updated_at,
        )

    @classmethod
    def from_row(cls, row: tuple) -> "Memory":
        (
            memory_id, memory_type, content, importance, confidence,
            sensitivity, source, tags_json, metadata_json,
            created_at, updated_at,
        ) = row
        return cls(
            id=memory_id,
            memory_type=MemoryType(memory_type),
            content=content,
            importance=importance,
            confidence=confidence,
            sensitivity=sensitivity,
            source=source,
            tags=json.loads(tags_json or "[]"),
            metadata=json.loads(metadata_json or "{}"),
            created_at=created_at,
            updated_at=updated_at,
        )
