"""
A.E.G.I.S. Memory Scoring.

Combines multiple signals to rank retrieved memories:

- lexical relevance
- importance
- confidence
- recency
- memory type

Lifecycle filtering remains the responsibility of the store/retriever.

The scorer does not modify memories.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from .models import Memory, MemoryType


class MemoryScorer:
    """
    Deterministic memory ranking system.

    The default weights intentionally favor query relevance while
    allowing important and reliable memories to rank higher.
    """

    DEFAULT_WEIGHTS = {
        "relevance": 0.55,
        "importance": 0.20,
        "confidence": 0.15,
        "recency": 0.10,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
    ):
        self.weights = (
            dict(weights)
            if weights is not None
            else dict(self.DEFAULT_WEIGHTS)
        )

        self._normalize_weights()

    def score(
        self,
        memory: Memory,
        query: str,
    ) -> float:
        """
        Calculate a final ranking score between 0.0 and 1.0.
        """

        relevance = self.relevance(
            memory,
            query,
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

        score = (
            relevance
            * self.weights["relevance"]
            + importance
            * self.weights["importance"]
            + confidence
            * self.weights["confidence"]
            + recency
            * self.weights["recency"]
        )

        return self._clamp(score)

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
            " ".join(memory.tags)
        )

        if not content_tokens and not tag_tokens:
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
        ) / len(query_tokens)

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
            token_score + phrase_bonus
        )

    def recency(
        self,
        memory: Memory,
        *,
        half_life_days: float = 30.0,
    ) -> float:
        """
        Calculate recency using exponential decay.

        A newly created memory approaches 1.0.
        Older memories gradually approach 0.0.

        half_life_days controls how quickly the score decays.
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
            age_seconds / 86400.0
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

    def rank(
        self,
        memories: list[Memory],
        query: str,
    ) -> list[Memory]:
        """
        Return memories sorted from highest score to lowest score.

        Ties are resolved deterministically using:
        1. importance
        2. confidence
        3. updated timestamp
        """

        scored = [
            (
                memory,
                self.score(
                    memory,
                    query,
                ),
            )
            for memory in memories
        ]

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

    def score_breakdown(
        self,
        memory: Memory,
        query: str,
    ) -> dict[str, float]:
        """
        Return individual scoring components.

        Useful for debugging and future observability.
        """

        return {
            "relevance": self.relevance(
                memory,
                query,
            ),
            "importance": self._clamp(
                memory.importance
            ),
            "confidence": self._clamp(
                memory.confidence
            ),
            "recency": self.recency(
                memory
            ),
            "final": self.score(
                memory,
                query,
            ),
        }

    # =========================================================
    # Helpers
    # =========================================================

    def _normalize_weights(self) -> None:
        total = sum(
            self.weights.values()
        )

        if total <= 0:
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

    @staticmethod
    def _tokens(
        text: str,
    ) -> set[str]:

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

        return " ".join(
            text.lower().split()
        )

    @staticmethod
    def _parse_datetime(
        value: str | None,
    ) -> datetime | None:

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

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )
    