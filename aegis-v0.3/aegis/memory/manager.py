"""
A.E.G.I.S. Memory Manager.

Provides the public API for:
- Explicit memory storage
- Automatic memory candidate storage
- Memory retrieval
- Recent memory listing
- Memory inspection
- Memory deletion
- Complete memory clearing
- Basic duplicate detection
- Memory formatting for the A.E.G.I.S. interface

v0.4.2c adds:
- Lexical similarity detection
- Duplicate candidate detection
- Governed automatic-memory storage
"""

from __future__ import annotations

import re

from .models import Memory, MemoryType
from .policy import MemoryPolicy
from .retrieval import MemoryRetriever
from .store import SQLiteMemoryStore


class MemoryManager:
    """
    Public API for A.E.G.I.S. persistent memory.

    MemoryManager is the boundary between the rest of A.E.G.I.S.
    and the underlying SQLite memory store.

    The LLM should never write directly to SQLite.
    All memory creation must pass through this class.
    """

    def __init__(
        self,
        database_path: str = "data/aegis_memory.db",
        policy: MemoryPolicy | None = None,
    ):
        self.store = SQLiteMemoryStore(database_path)

        self.policy = (
            policy
            if policy is not None
            else MemoryPolicy()
        )

        self.retriever = MemoryRetriever(
            self.store
        )

    # =========================================================
    # BASIC MEMORY STORAGE
    # =========================================================

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
        """
        Store a memory after passing it through MemoryPolicy.

        This is used for both explicit and controlled automatic
        memory creation.

        Returns:
            Memory object when stored.
            None when rejected by policy or empty.
        """

        content = content.strip()

        if not content:
            return None

        memory = Memory(
            content=content,
            memory_type=memory_type,
            importance=self._clamp(importance),
            confidence=self._clamp(confidence),
            sensitivity=self._clamp(sensitivity),
            source=source,
            tags=tags or [],
            metadata=metadata or {},
        )

        if not self.policy.should_store(
            memory,
            explicit=explicit,
        ):
            return None

        return self.store.save(memory)

    # =========================================================
    # MEMORY RETRIEVAL
    # =========================================================

    def recall(
        self,
        query: str,
        limit: int = 8,
    ) -> list[Memory]:
        """
        Retrieve memories relevant to a query.
        """

        query = query.strip()

        if not query:
            return []

        return self.retriever.retrieve(
            query,
            limit=limit,
        )

    def recent(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        """
        Return recently stored memories.

        Optionally filters by MemoryType.
        """

        return self.store.list(
            memory_type=memory_type,
            limit=limit,
        )

    def get(
        self,
        memory_id: str,
    ) -> Memory | None:
        """
        Retrieve a single memory by ID.
        """

        memory_id = memory_id.strip()

        if not memory_id:
            return None

        return self.store.get(
            memory_id
        )

    # =========================================================
    # MEMORY DELETION
    # =========================================================

    def forget(
        self,
        memory_id: str,
    ) -> bool:
        """
        Delete one memory by ID.

        Returns:
            True if the memory was deleted.
            False if no matching memory existed.
        """

        memory_id = memory_id.strip()

        if not memory_id:
            return False

        return self.store.delete(
            memory_id
        )

    def clear(self) -> int:
        """
        Delete every stored memory.

        Returns:
            Number of deleted memories.
        """

        return self.store.clear()

    # =========================================================
    # MEMORY INSPECTION
    # =========================================================

    def format_memory(
        self,
        memory: Memory,
    ) -> str:
        """
        Convert one memory into a human-readable inspection view.
        """

        tags = (
            ", ".join(memory.tags)
            if memory.tags
            else "None"
        )

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
        """
        Format relevant memories for injection into the LLM context.
        """

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

    # =========================================================
    # DUPLICATE DETECTION
    # =========================================================

    @staticmethod
    def _normalize_text(
        text: str,
    ) -> set[str]:
        """
        Convert text into normalized tokens.

        This is intentionally simple for v0.4.2c.
        Semantic embeddings will be introduced later.
        """

        return {
            token
            for token in re.findall(
                r"\b[a-zA-Z0-9_]+\b",
                text.lower(),
            )
            if len(token) > 2
        }

    @classmethod
    def _similarity(
        cls,
        first: str,
        second: str,
    ) -> float:
        """
        Calculate Jaccard token similarity.

        Formula:

            intersection / union

        Returns:
            Value between 0.0 and 1.0.
        """

        first_tokens = cls._normalize_text(
            first
        )

        second_tokens = cls._normalize_text(
            second
        )

        if not first_tokens or not second_tokens:
            return 0.0

        intersection = (
            first_tokens
            & second_tokens
        )

        union = (
            first_tokens
            | second_tokens
        )

        return len(intersection) / len(
            union
        )

    def find_similar(
        self,
        content: str,
        threshold: float = 0.60,
        limit: int = 10,
    ) -> list[tuple[Memory, float]]:
        """
        Find memories similar to a candidate.

        Args:
            content:
                Candidate memory text.

            threshold:
                Minimum similarity required.

            limit:
                Maximum number of matches.

        Returns:
            List of:
                (Memory, similarity_score)

        Results are sorted from most similar to least similar.
        """

        content = content.strip()

        if not content:
            return []

        memories = self.recent(
            limit=1000
        )

        matches: list[
            tuple[Memory, float]
        ] = []

        for memory in memories:
            similarity = self._similarity(
                content,
                memory.content,
            )

            if similarity >= threshold:
                matches.append(
                    (
                        memory,
                        similarity,
                    )
                )

        matches.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        return matches[:limit]

    # =========================================================
    # AUTOMATIC MEMORY GOVERNANCE
    # =========================================================

    def store_candidate(
        self,
        content: str,
        memory_type: MemoryType,
        *,
        importance: float,
        confidence: float,
        sensitivity: float,
        source: str = "automatic_extraction",
        tags: list[str] | None = None,
    ) -> tuple[str, Memory | None]:
        """
        Process an automatically extracted memory candidate.

        The candidate passes through:

            Candidate
                ↓
            Duplicate check
                ↓
            MemoryPolicy
                ↓
            MemoryManager.remember()
                ↓
            SQLite

        Returns:

            ("stored", Memory)
                Candidate was accepted and stored.

            ("duplicate", Memory)
                Candidate matched an existing memory.

            ("rejected", None)
                Candidate failed MemoryPolicy.
        """

        content = content.strip()

        if not content:
            return (
                "rejected",
                None,
            )

        # -----------------------------------------------------
        # 1. Clamp model-generated scores.
        # -----------------------------------------------------

        importance = self._clamp(
            importance
        )

        confidence = self._clamp(
            confidence
        )

        sensitivity = self._clamp(
            sensitivity
        )

        # -----------------------------------------------------
        # Automatic sensitivity gate.
        #
        # Automatically extracted sensitive information must
        # never be silently persisted.
        #
        # Explicit /memory remember commands use remember()
        # directly and therefore do not come through this path.
        # -----------------------------------------------------

        if sensitivity > 0:
            return (
                "rejected",
                None,
            )

        # -----------------------------------------------------
        # 2. Check for an existing memory.
        # -----------------------------------------------------

        similar = self.find_similar(
            content,
            threshold=0.60,
        )

        if similar:
            existing, similarity = (
                similar[0]
            )

            # Strong similarity means this is probably
            # the same piece of information.
            if similarity >= 0.80:
                return (
                    "duplicate",
                    existing,
                )

        # -----------------------------------------------------
        # 3. Send the candidate through the normal policy.
        # -----------------------------------------------------

        memory = self.remember(
            content,
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            sensitivity=sensitivity,
            source=source,
            tags=tags,
            explicit=False,
        )

        # -----------------------------------------------------
        # 4. Policy rejected the candidate.
        # -----------------------------------------------------

        if memory is None:
            return (
                "rejected",
                None,
            )

        # -----------------------------------------------------
        # 5. Candidate successfully stored.
        # -----------------------------------------------------

        return (
            "stored",
            memory,
        )

    # =========================================================
    # UTILITIES
    # =========================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        """
        Clamp a numerical score to [0.0, 1.0].
        """

        try:
            value = float(value)
        except (
            TypeError,
            ValueError,
        ):
            return 0.0

        return max(
            0.0,
            min(
                1.0,
                value,
            ),
        )