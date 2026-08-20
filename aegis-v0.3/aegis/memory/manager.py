from __future__ import annotations

from .models import Memory, MemoryType
from .policy import MemoryPolicy
from .retrieval import MemoryRetriever
from .store import SQLiteMemoryStore


class MemoryManager:
    """Public API for A.E.G.I.S. persistent memory."""

    def __init__(
        self,
        database_path: str = "data/aegis_memory.db",
        policy: MemoryPolicy | None = None,
    ):
        self.store = SQLiteMemoryStore(database_path)
        self.policy = policy or MemoryPolicy()
        self.retriever = MemoryRetriever(self.store)

    def remember(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODIC,
        *,
        importance: float = 0.5,
        confidence: float = 1.0,
        sensitivity: float = 0.0,
        source: str = "conversation",
        tags: list[str] | None = None,
        metadata: dict | None = None,
        explicit: bool = False,
    ) -> Memory | None:

        memory = Memory(
            content=content.strip(),
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            sensitivity=sensitivity,
            source=source,
            tags=tags or [],
            metadata=metadata or {},
        )

        if not memory.content:
            return None

        if not self.policy.should_store(
            memory,
            explicit=explicit,
        ):
            return None

        return self.store.save(memory)

    def recall(
        self,
        query: str,
        limit: int = 8,
    ) -> list[Memory]:
        return self.retriever.retrieve(
            query,
            limit=limit,
        )

    def recent(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        return self.store.list(
            memory_type=memory_type,
            limit=limit,
        )

    def get(
        self,
        memory_id: str,
    ) -> Memory | None:
        """Retrieve one memory by ID."""

        return self.store.get(memory_id)

    def forget(
        self,
        memory_id: str,
    ) -> bool:
        """Delete one memory."""

        return self.store.delete(memory_id)

    def clear(self) -> int:
        """Delete every stored memory."""

        return self.store.clear()

    def format_memory(
        self,
        memory: Memory,
    ) -> str:
        """Create a human-readable inspection view."""

        tags = ", ".join(memory.tags) if memory.tags else "None"

        return (
            f"Memory ID: {memory.id}\n"
            f"Type: {memory.memory_type.value}\n"
            f"Content: {memory.content}\n"
            f"Importance: {memory.importance:.2f}\n"
            f"Confidence: {memory.confidence:.2f}\n"
            f"Sensitivity: {memory.sensitivity:.2f}\n"
            f"Source: {memory.source}\n"
            f"Tags: {tags}\n"
            f"Created: {memory.created_at}\n"
            f"Updated: {memory.updated_at}"
        )

    def format_context(
        self,
        query: str,
        limit: int = 8,
    ) -> str:

        memories = self.recall(
            query,
            limit=limit,
        )

        if not memories:
            return "No relevant memories were found."

        lines = [
            "Relevant A.E.G.I.S. memories:"
        ]

        for memory in memories:
            lines.append(
                f"- [{memory.memory_type.value}] "
                f"{memory.content}"
            )

        return "\n".join(lines)