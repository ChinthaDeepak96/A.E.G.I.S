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

        if not self.policy.should_store(memory, explicit=explicit):
            return None

        return self.store.save(memory)

    def recall(self, query: str, limit: int = 8) -> list[Memory]:
        return self.retriever.retrieve(query, limit=limit)

    def recent(self, memory_type: MemoryType | None = None, limit: int = 20) -> list[Memory]:
        return self.store.list(memory_type=memory_type, limit=limit)

    def forget(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    def clear(self) -> int:
        return self.store.clear()

    def format_context(self, query: str, limit: int = 8) -> str:
        memories = self.recall(query, limit=limit)
        if not memories:
            return "No relevant memories were found."

        lines = ["Relevant A.E.G.I.S. memories:"]
        lines.extend(f"- [{m.memory_type.value}] {m.content}" for m in memories)
        return "\n".join(lines)
