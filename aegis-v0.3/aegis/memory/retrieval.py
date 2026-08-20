"""
A.E.G.I.S. Memory Retrieval.

Retrieval happens in two stages:

    1. SQLite performs inexpensive lexical candidate discovery.
    2. MemoryScorer ranks those candidates using multiple signals.

The public retrieve() API remains unchanged.
"""

from __future__ import annotations

from .models import Memory
from .scoring import MemoryScorer
from .store import SQLiteMemoryStore


class MemoryRetriever:
    """
    Two-stage memory retrieval.

    Stage 1:
        Lexical candidate discovery through SQLiteMemoryStore.

    Stage 2:
        Multi-signal ranking through MemoryScorer.
    """

    def __init__(
        self,
        store: SQLiteMemoryStore,
        scorer: MemoryScorer | None = None,
    ):
        self.store = store

        self.scorer = (
            scorer
            if scorer is not None
            else MemoryScorer()
        )

    def retrieve(
        self,
        query: str,
        limit: int = 8,
    ) -> list[Memory]:
        """
        Retrieve and rank memories relevant to a query.
        """

        query = query.strip()

        if not query:
            return []

        limit = max(
            1,
            int(limit),
        )

        # Retrieve a larger candidate pool first so that scoring
        # has enough memories to rank meaningfully.
        candidate_limit = max(
            limit * 3,
            20,
        )

        candidates = self.store.search_text(
            query,
            limit=candidate_limit,
        )

        if not candidates:
            return []

        ranked = self.scorer.rank(
            candidates,
            query,
        )

        return ranked[:limit]