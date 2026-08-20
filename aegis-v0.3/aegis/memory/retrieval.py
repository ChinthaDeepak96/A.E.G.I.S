"""
A.E.G.I.S. Memory Retrieval.

Hybrid retrieval combines:

    1. SQLite lexical candidate discovery.
    2. Persisted semantic embedding candidate discovery.
    3. Candidate fusion.
    4. Multi-signal ranking.
    5. Final relevance filtering.
    6. Retrieval usage tracking.

Usage supports two modes:

    1. In-memory mode
       MemoryUsage objects remain in the retriever.

    2. Persistent mode
       SQLiteMemoryUsageStore persists usage statistics.

The retriever does not modify Memory objects.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import MemoryEmbedder
from .models import Memory, MemoryStatus
from .scoring import MemoryScorer
from .store import SQLiteMemoryStore
from .usage import MemoryUsage
from .usage_store import SQLiteMemoryUsageStore


@dataclass(frozen=True)
class RetrievalCandidate:
    """
    Internal immutable retrieval candidate.

    A candidate can originate from:

        - lexical retrieval
        - semantic retrieval
        - both
    """

    memory: Memory

    lexical_score: float = 0.0

    semantic_score: float = 0.0


class MemoryRetriever:
    """
    Hybrid lexical + semantic memory retriever.

    Retrieval pipeline:

        query
          │
          ├── SQLite lexical discovery
          │
          └── semantic embedding discovery
                    │
                    ▼
              candidate fusion
                    │
                    ▼
               hybrid ranking
                    │
                    ▼
             final relevance gate
                    │
                    ▼
              returned memories
                    │
                    ▼
              usage tracking
                    │
                    ▼
        optional SQLite persistence
    """

    # =========================================================
    # RETRIEVAL CONFIGURATION
    # =========================================================

    # Candidate discovery threshold.
    #
    # This is deliberately permissive because semantic retrieval
    # is used to build the candidate pool.
    DEFAULT_SEMANTIC_THRESHOLD = 0.20

    # Final semantic admission threshold.
    #
    # A semantic-only candidate must meet this stronger threshold
    # before it can actually be returned.
    DEFAULT_MIN_SEMANTIC_SCORE = 0.35

    DEFAULT_LEXICAL_WEIGHT = 0.45

    DEFAULT_SEMANTIC_WEIGHT = 0.55

    # =========================================================
    # INITIALIZATION
    # =========================================================

    def __init__(
        self,
        store: SQLiteMemoryStore,
        scorer: MemoryScorer | None = None,
        embedder: MemoryEmbedder | None = None,
        usage: dict[str, MemoryUsage] | None = None,
        usage_store: (
            SQLiteMemoryUsageStore | None
        ) = None,
        *,
        semantic_threshold: float = (
            DEFAULT_SEMANTIC_THRESHOLD
        ),
        min_semantic_score: float = (
            DEFAULT_MIN_SEMANTIC_SCORE
        ),
        lexical_weight: float = (
            DEFAULT_LEXICAL_WEIGHT
        ),
        semantic_weight: float = (
            DEFAULT_SEMANTIC_WEIGHT
        ),
    ):
        """
        Initialize the memory retriever.

        Parameters
        ----------
        store:
            Main SQLite memory store.

        scorer:
            Memory ranking/scoring implementation.

        embedder:
            Semantic embedding implementation.

        usage:
            Optional in-memory usage registry.

        usage_store:
            Optional persistent usage store.

        semantic_threshold:
            Minimum semantic similarity required for candidate
            discovery.

        min_semantic_score:
            Minimum semantic similarity required for final
            admission of semantic-only candidates.

        lexical_weight:
            Relative weight of lexical relevance.

        semantic_weight:
            Relative weight of semantic relevance.
        """

        self.store = store

        self.scorer = (
            scorer
            if scorer is not None
            else MemoryScorer()
        )

        self.embedder = (
            embedder
            if embedder is not None
            else MemoryEmbedder()
        )

        # -----------------------------------------------------
        # In-memory usage registry.
        #
        # This remains available for existing callers and tests
        # that do not configure persistent usage.
        # -----------------------------------------------------

        self.usage = (
            usage
            if usage is not None
            else {}
        )

        # -----------------------------------------------------
        # Optional persistent usage store.
        #
        # When configured, persistent usage takes precedence
        # over the in-memory registry.
        # -----------------------------------------------------

        self.usage_store = usage_store

        self.semantic_threshold = max(
            0.0,
            min(
                1.0,
                float(
                    semantic_threshold
                ),
            ),
        )

        self.min_semantic_score = max(
            0.0,
            min(
                1.0,
                float(
                    min_semantic_score
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
        Retrieve memories using hybrid lexical + semantic
        retrieval.

        Only candidates passing the final relevance gate are
        returned.

        Usage is recorded only for memories actually returned.

        Empty queries produce no retrieval and no usage event.
        """

        query = query.strip()

        if not query:
            return []

        limit = max(
            1,
            int(limit),
        )

        # -----------------------------------------------------
        # Retrieve a larger candidate pool than the requested
        # result count so ranking has enough candidates.
        # -----------------------------------------------------

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

        results: list[Memory] = []

        for candidate in ranked:

            if not self._is_retrievable(
                candidate
            ):
                continue

            results.append(
                candidate.memory
            )

            if len(results) >= limit:
                break

        # -----------------------------------------------------
        # No valid final results means no usage event.
        # -----------------------------------------------------

        if not results:
            return []

        # -----------------------------------------------------
        # Record usage only after final admission.
        # -----------------------------------------------------

        self._record_retrievals(
            results
        )

        return results

    # =========================================================
    # USAGE
    # =========================================================

    def get_usage(
        self,
        memory_id: str,
    ) -> MemoryUsage:
        """
        Return usage statistics for a memory.

        Persistent storage is preferred when configured.

        Otherwise usage is stored in the in-memory registry.

        If no usage record exists, a zeroed record is created.
        """

        memory_id = memory_id.strip()

        if not memory_id:
            raise ValueError(
                "memory_id cannot be empty"
            )

        # -----------------------------------------------------
        # Persistent usage mode.
        # -----------------------------------------------------

        if self.usage_store is not None:

            return self.usage_store.get_or_create(
                memory_id
            )

        # -----------------------------------------------------
        # In-memory usage mode.
        # -----------------------------------------------------

        return self.usage.setdefault(
            memory_id,
            MemoryUsage(),
        )

    def _record_retrievals(
        self,
        memories: list[Memory],
    ) -> None:
        """
        Record one retrieval event for every returned memory.

        In persistent mode:

            load → mutate → save

        In-memory mode:

            mutate registry directly.
        """

        for memory in memories:

            # -------------------------------------------------
            # Persistent mode.
            # -------------------------------------------------

            if self.usage_store is not None:

                usage = (
                    self.usage_store.get_or_create(
                        memory.id
                    )
                )

                usage.record_retrieval()

                self.usage_store.save(
                    memory.id,
                    usage,
                )

                continue

            # -------------------------------------------------
            # In-memory mode.
            # -------------------------------------------------

            usage = self.usage.setdefault(
                memory.id,
                MemoryUsage(),
            )

            usage.record_retrieval()

    # =========================================================
    # LEXICAL RETRIEVAL
    # =========================================================

    def lexical_candidates(
        self,
        query: str,
        limit: int = 20,
    ) -> list[Memory]:
        """
        Return lexical candidates from SQLite.
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
        Return semantic candidates as:

            (memory, cosine_similarity)

        Persisted embeddings are used directly.

        The semantic threshold here is only for candidate
        discovery. Final admission is handled by
        _is_retrievable().
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

            # -------------------------------------------------
            # Ignore embeddings generated using another model.
            # -------------------------------------------------

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

        # -----------------------------------------------------
        # Strongest semantic candidates first.
        # -----------------------------------------------------

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
        Merge lexical and semantic candidate sets.

        A memory appearing in both sources becomes one candidate
        containing both relevance signals.
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
        # Lexical candidates.
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
        # Semantic candidates.
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

            # -------------------------------------------------
            # Candidate exists in both retrieval systems.
            # Preserve both signals.
            # -------------------------------------------------

            candidates[
                memory.id
            ] = RetrievalCandidate(
                memory=existing.memory,
                lexical_score=(
                    existing.lexical_score
                ),
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

            1. lexical relevance
            2. semantic relevance
            3. importance
            4. confidence
            5. recency

        Semantic scores are reused directly.

        No embedding is regenerated during ranking.
        """

        scored: list[
            tuple[
                RetrievalCandidate,
                float,
            ]
        ] = []

        for candidate in candidates:

            # -------------------------------------------------
            # Combine lexical and semantic relevance.
            # -------------------------------------------------

            hybrid_relevance = (
                candidate.lexical_score
                * self.lexical_weight
                + candidate.semantic_score
                * self.semantic_weight
            )

            # -------------------------------------------------
            # Existing memory quality score.
            # -------------------------------------------------

            memory_score = (
                self.scorer.score(
                    candidate.memory,
                    query,
                    semantic_score=(
                        candidate.semantic_score
                    ),
                )
            )

            # -------------------------------------------------
            # Final retrieval score.
            #
            # 60% hybrid relevance
            # 40% memory quality
            # -------------------------------------------------

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

        # -----------------------------------------------------
        # Deterministic ranking.
        # -----------------------------------------------------

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
    # FINAL RETRIEVAL GATE
    # =========================================================

    def _is_retrievable(
        self,
        candidate: RetrievalCandidate,
    ) -> bool:
        """
        Determine whether a candidate has enough evidence to be
        returned.

        A lexical match is accepted when meaningful lexical
        relevance exists.

        A semantic-only candidate must satisfy the stronger
        final semantic threshold.

        This prevents weak semantic candidates from escaping the
        candidate pool and becoming false-positive memories.
        """

        # -----------------------------------------------------
        # Lexical path.
        # -----------------------------------------------------

        if (
            candidate.lexical_score
            > 0.0
        ):
            return True

        # -----------------------------------------------------
        # Semantic-only path.
        # -----------------------------------------------------

        return (
            candidate.semantic_score
            >= self.min_semantic_score
        )

    # =========================================================
    # LEXICAL SCORING
    # =========================================================

    def _lexical_relevance(
        self,
        memory: Memory,
        query: str,
    ) -> float:
        """
        Calculate lexical relevance using MemoryScorer.
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
        Normalize lexical and semantic weights so that:

            lexical_weight + semantic_weight == 1.0
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