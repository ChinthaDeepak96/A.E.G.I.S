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


class MemoryStatus(str, Enum):
    """
    Lifecycle state of a persistent memory.
    """

    NEW = "new"
    ACTIVE = "active"
    STALE = "stale"
    ARCHIVED = "archived"
    SUPERSEDED = "superseded"


@dataclass
class Memory:
    """
    Persistent A.E.G.I.S. memory.

    A memory has both:
    - cognitive metadata: importance, confidence, sensitivity
    - lifecycle metadata: status, timestamps

    Lifecycle status determines whether the memory should normally
    participate in retrieval.
    """

    content: str

    memory_type: MemoryType = MemoryType.EPISODIC

    importance: float = 0.5
    confidence: float = 1.0
    sensitivity: float = 0.0

    source: str = "conversation"

    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    status: MemoryStatus = MemoryStatus.ACTIVE

    id: str = field(
        default_factory=lambda: str(uuid.uuid4())
    )

    created_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    updated_at: str = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        ).isoformat()
    )

    stale_at: str | None = None
    archived_at: str | None = None
    superseded_by: str | None = None

    def __post_init__(self) -> None:

        self.importance = max(
            0.0,
            min(
                1.0,
                float(self.importance),
            ),
        )

        self.confidence = max(
            0.0,
            min(
                1.0,
                float(self.confidence),
            ),
        )

        self.sensitivity = max(
            0.0,
            min(
                1.0,
                float(self.sensitivity),
            ),
        )

        if not isinstance(
            self.memory_type,
            MemoryType,
        ):
            self.memory_type = MemoryType(
                self.memory_type
            )

        if not isinstance(
            self.status,
            MemoryStatus,
        ):
            self.status = MemoryStatus(
                self.status
            )

    def touch(self) -> None:
        """
        Update the modification timestamp.
        """

        self.updated_at = datetime.now(
            timezone.utc
        ).isoformat()

    def activate(self) -> None:
        """
        Mark the memory as currently active.
        """

        self.status = MemoryStatus.ACTIVE
        self.stale_at = None
        self.archived_at = None
        self.superseded_by = None
        self.touch()

    def mark_stale(self) -> None:
        """
        Mark the memory as stale without deleting it.
        """

        self.status = MemoryStatus.STALE

        self.stale_at = datetime.now(
            timezone.utc
        ).isoformat()

        self.touch()

    def archive(self) -> None:
        """
        Permanently remove the memory from normal retrieval
        while retaining it in storage.
        """

        self.status = MemoryStatus.ARCHIVED

        self.archived_at = datetime.now(
            timezone.utc
        ).isoformat()

        self.touch()

    def supersede(
        self,
        replacement_memory_id: str,
    ) -> None:
        """
        Mark this memory as replaced by another memory.
        """

        self.status = MemoryStatus.SUPERSEDED

        self.superseded_by = (
            replacement_memory_id
        )

        self.touch()

    def to_record(self) -> tuple:
        """
        Convert the memory to the current SQLite record format.
        """

        return (
            self.id,
            self.memory_type.value,
            self.content,
            self.importance,
            self.confidence,
            self.sensitivity,
            self.source,
            json.dumps(
                self.tags,
                ensure_ascii=False,
            ),
            json.dumps(
                self.metadata,
                ensure_ascii=False,
            ),
            self.status.value,
            self.created_at,
            self.updated_at,
            self.stale_at,
            self.archived_at,
            self.superseded_by,
        )

    @classmethod
    def from_row(
        cls,
        row: tuple,
    ) -> "Memory":
        """
        Reconstruct a Memory from a SQLite row.

        Expected current schema:

        id
        memory_type
        content
        importance
        confidence
        sensitivity
        source
        tags_json
        metadata_json
        status
        created_at
        updated_at
        stale_at
        archived_at
        superseded_by
        """

        (
            memory_id,
            memory_type,
            content,
            importance,
            confidence,
            sensitivity,
            source,
            tags_json,
            metadata_json,
            status,
            created_at,
            updated_at,
            stale_at,
            archived_at,
            superseded_by,
        ) = row

        return cls(
            id=memory_id,
            memory_type=MemoryType(
                memory_type
            ),
            content=content,
            importance=importance,
            confidence=confidence,
            sensitivity=sensitivity,
            source=source,
            tags=json.loads(
                tags_json or "[]"
            ),
            metadata=json.loads(
                metadata_json or "{}"
            ),
            status=MemoryStatus(
                status
            ),
            created_at=created_at,
            updated_at=updated_at,
            stale_at=stale_at,
            archived_at=archived_at,
            superseded_by=superseded_by,
        )