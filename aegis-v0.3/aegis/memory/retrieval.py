"""
A.E.G.I.S. Memory Retrieval.

Hybrid retrieval combines:

    1. SQLite lexical candidate discovery.
    2. Persisted semantic embedding candidate discovery.
    3. Candidate fusion.
    4. Multi-signal ranking.

Semantic similarity is calculated once per candidate and reused
during final scoring.

The retriever does not modify memories.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import MemoryEmbedder
from .models import Memory, MemoryStatus
from .scoring import MemoryScorer
from .store import SQLiteMemoryStore


@dataclass(frozen=True)
class RetrievalCandidate:
    """
    Internal immutable retrieval candidate.
    """

    memory: Memory

    lexical_score: float = 0.0

    semantic_score: float = 0.0


class MemoryRetriever:
    """
    Hybrid lexical + semantic memory retriever.
    """

    DEFAULT_SEMANTIC_THRESHOLD = 0.20

    DEFAULT_LEXICAL_WEIGHT = 0.45

    DEFAULT_SEMANTIC_WEIGHT = 0.55

    def __init__(
        self,
        store: SQLiteMemoryStore,
        scorer: MemoryScorer | None = None,
        embedder: MemoryEmbedder | None = None,
        *,
        semantic_threshold: float = (
            DEFAULT_SEMANTIC_THRESHOLD
        ),
        lexical_weight: float = (
            DEFAULT_LEXICAL_WEIGHT
        ),
        semantic_weight: float = (
            DEFAULT_SEMANTIC_WEIGHT
        ),
    ):
        self.store = store

        self.embedder = (
            embedder
            if embedder is not None
            else MemoryEmbedder()
        )

        self.scorer = (
            scorer
            if scorer is not None
            else MemoryScorer()
        )

        self.semantic_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    semantic_threshold
                ),
            ),
        )

        self.lexical_weight = max(
            0.0,
            float(
                lexical_weight
            ),
        )

        self.semantic_weight = max(
            0.0,
            float(
                semantic_weight
            ),
        )

        self._normalize_weights()

    # =========================================================
    # PUBLIC RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        query: str,
        limit: int = 8,
    ) -> list[Memory]:
        """
        Retrieve memories using hybrid retrieval.
        """

        query = query.strip()

        if not query:
            return []

        limit = max(
            1,
            int(limit),
        )

        candidate_limit = max(
            limit * 4,
            20,
        )

        candidates = self._collect_candidates(
            query,
            candidate_limit,
        )

        if not candidates:
            return []

        ranked = self._rank_candidates(
            candidates,
            query,
        )

        return [
            candidate.memory
            for candidate in ranked[:limit]
        ]

    # =========================================================
    # LEXICAL RETRIEVAL
    # =========================================================

    def lexical_candidates(
        self,
        query: str,
        limit: int = 20,
    ) -> list[Memory]:
        """
        Return lexical candidates.
        """

        query = query.strip()

        if not query:
            return []

        return self.store.search_text(
            query,
            limit=max(
                1,
                int(limit),
            ),
        )

    # =========================================================
    # SEMANTIC RETRIEVAL
    # =========================================================

    def semantic_candidates(
        self,
        query: str,
        limit: int = 20,
    ) -> list[
        tuple[
            Memory,
            float,
        ]
    ]:
        """
        Return:

            (memory, cosine_similarity)

        for semantic candidates.

        Persisted embeddings are used directly.
        """

        query = query.strip()

        if not query:
            return []

        limit = max(
            1,
            int(limit),
        )

        query_embedding = (
            self.embedder.encode(
                query
            )
        )

        memories = self.store.list(
            status=MemoryStatus.ACTIVE,
            limit=1000,
        )

        results: list[
            tuple[
                Memory,
                float,
            ]
        ] = []

        for memory in memories:

            stored = (
                self.store.get_embedding(
                    memory.id
                )
            )

            if stored is None:
                continue

            (
                embedding,
                model_name,
                dimensions,
                _created_at,
            ) = stored

            if (
                model_name
                != self.embedder.model_name
            ):
                continue

            if (
                dimensions
                != query_embedding.size
            ):
                continue

            try:
                similarity = (
                    self.embedder.vector_similarity(
                        query_embedding,
                        embedding,
                    )
                )

            except ValueError:
                continue

            if (
                similarity
                >= self.semantic_threshold
            ):
                results.append(
                    (
                        memory,
                        similarity,
                    )
                )

        results.sort(
            key=lambda item: (
                item[1],
                item[0].importance,
                item[0].confidence,
                item[0].updated_at,
            ),
            reverse=True,
        )

        return results[:limit]

    # =========================================================
    # CANDIDATE COLLECTION
    # =========================================================

    def _collect_candidates(
        self,
        query: str,
        limit: int,
    ) -> list[RetrievalCandidate]:
        """
        Merge lexical and semantic candidates.

        A memory appearing in both sources becomes one candidate
        containing both scores.
        """

        lexical = self.lexical_candidates(
            query,
            limit=limit,
        )

        semantic = self.semantic_candidates(
            query,
            limit=limit,
        )

        candidates: dict[
            str,
            RetrievalCandidate,
        ] = {}

        # -----------------------------------------------------
        # Lexical candidates
        # -----------------------------------------------------

        for memory in lexical:

            lexical_score = (
                self._lexical_relevance(
                    memory,
                    query,
                )
            )

            candidates[
                memory.id
            ] = RetrievalCandidate(
                memory=memory,
                lexical_score=lexical_score,
                semantic_score=0.0,
            )

        # -----------------------------------------------------
        # Semantic candidates
        # -----------------------------------------------------

        for (
            memory,
            semantic_score,
        ) in semantic:

            existing = candidates.get(
                memory.id
            )

            if existing is None:

                candidates[
                    memory.id
                ] = RetrievalCandidate(
                    memory=memory,
                    lexical_score=0.0,
                    semantic_score=semantic_score,
                )

                continue

            candidates[
                memory.id
            ] = RetrievalCandidate(
                memory=existing.memory,
                lexical_score=existing.lexical_score,
                semantic_score=semantic_score,
            )

        return list(
            candidates.values()
        )

    # =========================================================
    # RANKING
    # =========================================================

    def _rank_candidates(
        self,
        candidates: list[RetrievalCandidate],
        query: str,
    ) -> list[RetrievalCandidate]:
        """
        Rank candidates using:

            lexical score
            semantic score
            importance
            confidence
            recency

        Semantic scores are reused directly. No embeddings are
        regenerated here.
        """

        scored: list[
            tuple[
                RetrievalCandidate,
                float,
            ]
        ] = []

        for candidate in candidates:

            hybrid_relevance = (
                candidate.lexical_score
                * self.lexical_weight
                + candidate.semantic_score
                * self.semantic_weight
            )

            memory_score = (
                self.scorer.score(
                    candidate.memory,
                    query,
                    semantic_score=(
                        candidate.semantic_score
                    ),
                )
            )

            final_score = (
                0.60
                * hybrid_relevance
                + 0.40
                * memory_score
            )

            final_score = max(
                0.0,
                min(
                    1.0,
                    final_score,
                ),
            )

            scored.append(
                (
                    candidate,
                    final_score,
                )
            )

        scored.sort(
            key=lambda item: (
                item[1],
                item[0].memory.importance,
                item[0].memory.confidence,
                item[0].memory.updated_at,
            ),
            reverse=True,
        )

        return [
            candidate
            for candidate, _score
            in scored
        ]

    # =========================================================
    # LEXICAL SCORE
    # =========================================================

    def _lexical_relevance(
        self,
        memory: Memory,
        query: str,
    ) -> float:
        """
        Calculate lexical relevance.
        """

        return max(
            0.0,
            min(
                1.0,
                self.scorer.relevance(
                    memory,
                    query,
                ),
            ),
        )

    # =========================================================
    # WEIGHT NORMALIZATION
    # =========================================================

    def _normalize_weights(
        self,
    ) -> None:
        """
        Normalize lexical and semantic weights.
        """

        total = (
            self.lexical_weight
            + self.semantic_weight
        )

        if total <= 0.0:

            self.lexical_weight = (
                self.DEFAULT_LEXICAL_WEIGHT
            )

            self.semantic_weight = (
                self.DEFAULT_SEMANTIC_WEIGHT
            )

            total = (
                self.lexical_weight
                + self.semantic_weight
            )

        self.lexical_weight /= total

        self.semantic_weight /= total