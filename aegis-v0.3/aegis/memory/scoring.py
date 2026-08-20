"""
A.E.G.I.S. Memory Scoring.

Combines multiple signals to rank retrieved memories:

- lexical relevance
- semantic relevance
- importance
- confidence
- recency

Semantic similarity may be supplied by the retrieval layer.

When a precomputed semantic score is supplied, the scorer
reuses it and does not generate another embedding.

When no semantic score is supplied, MemoryScorer can lazily
use MemoryEmbedder for standalone semantic scoring.

The scorer does not access SQLite and does not modify memories.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Callable

from .embeddings import MemoryEmbedder
from .models import Memory


SemanticProvider = Callable[
    [Memory, str],
    float,
]


class MemoryScorer:
    """
    Deterministic memory ranking system.

    The scorer combines:

        lexical relevance
        semantic relevance
        importance
        confidence
        recency

    Semantic relevance can be:

        1. supplied directly to score() / rank()
        2. calculated through a semantic provider
        3. calculated through MemoryEmbedder

    The embedding model is loaded lazily.
    """

    DEFAULT_WEIGHTS = {
        "relevance": 0.45,
        "importance": 0.20,
        "confidence": 0.15,
        "recency": 0.10,
        "semantic": 0.10,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        semantic_provider: SemanticProvider | None = None,
        embedder: MemoryEmbedder | None = None,
    ):
        self.weights = (
            dict(weights)
            if weights is not None
            else dict(self.DEFAULT_WEIGHTS)
        )

        self.embedder = (
            embedder
            if embedder is not None
            else MemoryEmbedder()
        )

        self.semantic_provider = (
            semantic_provider
        )

        self._normalize_weights()

    # =========================================================
    # FINAL SCORE
    # =========================================================

    def score(
        self,
        memory: Memory,
        query: str,
        *,
        semantic_score: float | None = None,
    ) -> float:
        """
        Calculate a final ranking score.

        Args:
            memory:
                Memory being scored.

            query:
                User query.

            semantic_score:
                Optional precomputed cosine similarity.

                When supplied, the embedding model is not
                called again.

        Returns:
            Score in the range [0.0, 1.0].
        """

        lexical_relevance = self.relevance(
            memory,
            query,
        )

        if semantic_score is None:
            semantic_relevance = (
                self.semantic_relevance(
                    memory,
                    query,
                )
            )
        else:
            semantic_relevance = (
                self._semantic_similarity_to_score(
                    semantic_score
                )
            )

        importance = self._clamp(
            memory.importance
        )

        confidence = self._clamp(
            memory.confidence
        )

        recency = self.recency(
            memory
        )

        # -----------------------------------------------------
        # Combine lexical and semantic relevance.
        #
        # Semantic also has its own explicit weight so that
        # semantic matches can remain useful even when lexical
        # overlap is weak.
        # -----------------------------------------------------

        combined_relevance = (
            lexical_relevance * 0.5
            + semantic_relevance * 0.5
        )

        score = (
            combined_relevance
            * self.weights["relevance"]
            + semantic_relevance
            * self.weights["semantic"]
            + importance
            * self.weights["importance"]
            + confidence
            * self.weights["confidence"]
            + recency
            * self.weights["recency"]
        )

        return self._clamp(
            score
        )

    # =========================================================
    # LEXICAL RELEVANCE
    # =========================================================

    def relevance(
        self,
        memory: Memory,
        query: str,
    ) -> float:
        """
        Calculate lexical relevance.

        Considers:

        - memory content
        - tags
        - exact token matches
        - phrase match

        This method remains lexical for backward compatibility.
        """

        query_tokens = self._tokens(
            query
        )

        if not query_tokens:
            return 0.0

        content_tokens = self._tokens(
            memory.content
        )

        tag_tokens = self._tokens(
            " ".join(
                memory.tags
            )
        )

        if (
            not content_tokens
            and not tag_tokens
        ):
            return 0.0

        content_matches = sum(
            1
            for token in query_tokens
            if token in content_tokens
        )

        tag_matches = sum(
            1
            for token in query_tokens
            if token in tag_tokens
        )

        token_score = (
            content_matches
            + tag_matches * 1.5
        ) / len(
            query_tokens
        )

        normalized_query = self._normalize(
            query
        )

        normalized_content = self._normalize(
            memory.content
        )

        phrase_bonus = (
            0.25
            if (
                normalized_query
                and normalized_query
                in normalized_content
            )
            else 0.0
        )

        return self._clamp(
            token_score
            + phrase_bonus
        )

    # =========================================================
    # SEMANTIC RELEVANCE
    # =========================================================

    def semantic_relevance(
        self,
        memory: Memory,
        query: str,
    ) -> float:
        """
        Calculate semantic relevance.

        Priority:

            1. custom semantic provider
            2. MemoryEmbedder

        The provider/embedder returns cosine similarity in
        approximately [-1.0, 1.0].

        Conversion:

            similarity <= 0.0  -> 0.0 relevance
            similarity >  0.0  -> similarity

        This prevents unrelated vectors with cosine similarity
        near zero from incorrectly becoming 0.5 relevance.
        """

        query = query.strip()

        if not query:
            return 0.0

        try:
            if self.semantic_provider is not None:
                similarity = float(
                    self.semantic_provider(
                        memory,
                        query,
                    )
                )
            else:
                similarity = float(
                    self.embedder.similarity(
                        memory.content,
                        query,
                    )
                )

        except (
            TypeError,
            ValueError,
            RuntimeError,
        ):
            return 0.0

        return self._semantic_similarity_to_score(
            similarity
        )

    @staticmethod
    def _semantic_similarity_to_score(
        similarity: float,
    ) -> float:
        """
        Convert cosine similarity into [0.0, 1.0].

        Values <= 0 are treated as no semantic relevance.
        """

        return max(
            0.0,
            min(
                1.0,
                float(similarity),
            ),
        )

    # =========================================================
    # RECENCY
    # =========================================================

    def recency(
        self,
        memory: Memory,
        *,
        half_life_days: float = 30.0,
    ) -> float:
        """
        Calculate recency using exponential decay.

        A newly updated memory approaches 1.0.

        Older memories gradually approach 0.0.
        """

        created = self._parse_datetime(
            memory.updated_at
            or memory.created_at
        )

        if created is None:
            return 0.0

        now = datetime.now(
            timezone.utc
        )

        age_seconds = max(
            0.0,
            (
                now - created
            ).total_seconds(),
        )

        age_days = (
            age_seconds
            / 86400.0
        )

        if half_life_days <= 0:
            return 0.0

        return self._clamp(
            math.pow(
                0.5,
                age_days
                / half_life_days,
            )
        )

    # =========================================================
    # RANKING
    # =========================================================

    def rank(
        self,
        memories: list[Memory],
        query: str,
        *,
        semantic_scores: dict[str, float]
        | None = None,
    ) -> list[Memory]:
        """
        Rank memories from highest score to lowest.

        semantic_scores maps:

            memory_id -> cosine similarity

        When supplied, the precomputed values are reused.

        Memories without a supplied semantic score fall back
        to normal semantic_relevance().
        """

        semantic_scores = (
            semantic_scores or {}
        )

        scored = []

        for memory in memories:
            precomputed = (
                semantic_scores.get(
                    memory.id
                )
            )

            score = self.score(
                memory,
                query,
                semantic_score=precomputed,
            )

            scored.append(
                (
                    memory,
                    score,
                )
            )

        scored.sort(
            key=lambda item: (
                item[1],
                item[0].importance,
                item[0].confidence,
                item[0].updated_at,
            ),
            reverse=True,
        )

        return [
            memory
            for memory, _score in scored
        ]

    # =========================================================
    # SCORE BREAKDOWN
    # =========================================================

    def score_breakdown(
        self,
        memory: Memory,
        query: str,
        *,
        semantic_score: float | None = None,
    ) -> dict[str, float]:
        """
        Return individual scoring components.

        If semantic_score is supplied, it is reused.
        """

        lexical = self.relevance(
            memory,
            query,
        )

        if semantic_score is None:
            semantic = self.semantic_relevance(
                memory,
                query,
            )
        else:
            semantic = (
                self._semantic_similarity_to_score(
                    semantic_score
                )
            )

        importance = self._clamp(
            memory.importance
        )

        confidence = self._clamp(
            memory.confidence
        )

        recency = self.recency(
            memory
        )

        final = self.score(
            memory,
            query,
            semantic_score=semantic,
        )

        return {
            "relevance": lexical,
            "lexical_relevance": lexical,
            "semantic_relevance": semantic,
            "importance": importance,
            "confidence": confidence,
            "recency": recency,
            "final": final,
        }

    # =========================================================
    # WEIGHT NORMALIZATION
    # =========================================================

    def _normalize_weights(
        self,
    ) -> None:
        """
        Normalize configured weights so they sum to 1.0.
        """

        supported = {
            "relevance",
            "importance",
            "confidence",
            "recency",
            "semantic",
        }

        self.weights = {
            key: float(
                self.weights.get(
                    key,
                    0.0,
                )
            )
            for key in supported
        }

        total = sum(
            self.weights.values()
        )

        if total <= 0.0:
            self.weights = dict(
                self.DEFAULT_WEIGHTS
            )

            total = sum(
                self.weights.values()
            )

        for key in self.weights:
            self.weights[key] = (
                self.weights[key]
                / total
            )

    # =========================================================
    # TOKENIZATION
    # =========================================================

    @staticmethod
    def _tokens(
        text: str,
    ) -> set[str]:
        """
        Convert text into normalized lexical tokens.
        """

        return {
            token
            for token in re.findall(
                r"[a-z0-9]+",
                text.lower(),
            )
            if len(token) > 1
        }

    @staticmethod
    def _normalize(
        text: str,
    ) -> str:
        """
        Normalize text for phrase comparison.
        """

        return " ".join(
            text.lower().split()
        )

    # =========================================================
    # DATETIME
    # =========================================================

    @staticmethod
    def _parse_datetime(
        value: str | None,
    ) -> datetime | None:
        """
        Parse an ISO datetime into UTC.
        """

        if not value:
            return None

        try:
            parsed = datetime.fromisoformat(
                value
            )

            if parsed.tzinfo is None:
                parsed = parsed.replace(
                    tzinfo=timezone.utc
                )

            return parsed.astimezone(
                timezone.utc
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    # =========================================================
    # CLAMP
    # =========================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        """
        Clamp a numerical value to [0.0, 1.0].
        """

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )