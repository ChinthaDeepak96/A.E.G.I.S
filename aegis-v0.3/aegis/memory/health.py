"""
A.E.G.I.S. Memory Health.

Provides a deterministic, read-only assessment of memory health.

This module does NOT modify memories or the database.

Health is calculated from currently available memory signals:

    - lifecycle status
    - age
    - importance
    - confidence
    - staleness metadata
    - archival metadata
    - supersession metadata

The purpose of this module is to identify memories that may
require future maintenance.

Actual lifecycle mutations remain the responsibility of
MemoryManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .models import Memory, MemoryStatus


class MemoryHealth(str, Enum):
    """
    Deterministic memory health classification.
    """

    HEALTHY = "healthy"

    AGING = "aging"

    STALE_CANDIDATE = "stale_candidate"

    ARCHIVAL_CANDIDATE = "archival_candidate"


@dataclass(frozen=True)
class MemoryHealthReport:
    """
    Immutable health report for a single memory.

    The report contains observations and recommendations only.

    No memory is modified.
    """

    memory_id: str

    health: MemoryHealth

    age_days: float

    importance: float

    confidence: float

    status: MemoryStatus

    stale: bool

    archived: bool

    superseded: bool

    health_score: float

    reason: str


class MemoryHealthAnalyzer:
    """
    Deterministic first-generation memory health analyzer.

    Conservative defaults:

        ACTIVE + recent + reliable
            -> HEALTHY

        ACTIVE + older
            -> AGING

        ACTIVE + very old or weak confidence
            -> STALE_CANDIDATE

        STALE
            -> ARCHIVAL_CANDIDATE

        ARCHIVED / SUPERSEDED
            -> ARCHIVAL_CANDIDATE

    This class never modifies a Memory.
    """

    DEFAULT_AGING_DAYS = 30.0

    DEFAULT_STALE_DAYS = 90.0

    DEFAULT_ARCHIVAL_DAYS = 180.0

    DEFAULT_STALE_CONFIDENCE = 0.35

    DEFAULT_ARCHIVAL_CONFIDENCE = 0.20

    def __init__(
        self,
        *,
        aging_days: float = DEFAULT_AGING_DAYS,
        stale_days: float = DEFAULT_STALE_DAYS,
        archival_days: float = DEFAULT_ARCHIVAL_DAYS,
        stale_confidence: float = (
            DEFAULT_STALE_CONFIDENCE
        ),
        archival_confidence: float = (
            DEFAULT_ARCHIVAL_CONFIDENCE
        ),
    ):
        if aging_days < 0:
            raise ValueError(
                "aging_days must be non-negative"
            )

        if stale_days < aging_days:
            raise ValueError(
                "stale_days must be greater than or equal "
                "to aging_days"
            )

        if archival_days < stale_days:
            raise ValueError(
                "archival_days must be greater than or equal "
                "to stale_days"
            )

        self.aging_days = float(
            aging_days
        )

        self.stale_days = float(
            stale_days
        )

        self.archival_days = float(
            archival_days
        )

        self.stale_confidence = self._clamp(
            stale_confidence
        )

        self.archival_confidence = self._clamp(
            archival_confidence
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
        memory: Memory,
        *,
        now: datetime | None = None,
    ) -> MemoryHealthReport:
        """
        Analyze one memory.

        The memory is never modified.
        """

        if not isinstance(
            memory,
            Memory,
        ):
            raise TypeError(
                "memory must be a Memory instance"
            )

        current_time = (
            now
            if now is not None
            else datetime.now(
                timezone.utc
            )
        )

        current_time = self._ensure_utc(
            current_time
        )

        age_days = self._age_days(
            memory,
            current_time,
        )

        importance = self._clamp(
            memory.importance
        )

        confidence = self._clamp(
            memory.confidence
        )

        health = self._classify(
            memory,
            age_days,
            confidence,
        )

        health_score = self._health_score(
            memory,
            age_days,
            importance,
            confidence,
        )

        reason = self._reason(
            memory,
            health,
            age_days,
            confidence,
        )

        return MemoryHealthReport(
            memory_id=memory.id,
            health=health,
            age_days=age_days,
            importance=importance,
            confidence=confidence,
            status=memory.status,
            stale=(
                memory.status
                == MemoryStatus.STALE
            ),
            archived=(
                memory.status
                == MemoryStatus.ARCHIVED
            ),
            superseded=(
                memory.status
                == MemoryStatus.SUPERSEDED
            ),
            health_score=health_score,
            reason=reason,
        )

    def analyze_many(
        self,
        memories: list[Memory],
        *,
        now: datetime | None = None,
    ) -> list[MemoryHealthReport]:
        """
        Analyze multiple memories.

        Results preserve the input order.
        """

        return [
            self.analyze(
                memory,
                now=now,
            )
            for memory in memories
        ]

    def classify(
        self,
        memory: Memory,
        *,
        now: datetime | None = None,
    ) -> MemoryHealth:
        """
        Return only the health classification.
        """

        return self.analyze(
            memory,
            now=now,
        ).health

    # =========================================================
    # CLASSIFICATION
    # =========================================================

    def _classify(
        self,
        memory: Memory,
        age_days: float,
        confidence: float,
    ) -> MemoryHealth:
        """
        Determine deterministic health state.
        """

        # -----------------------------------------------------
        # Already archived or superseded.
        # -----------------------------------------------------

        if memory.status in {
            MemoryStatus.ARCHIVED,
            MemoryStatus.SUPERSEDED,
        }:
            return MemoryHealth.ARCHIVAL_CANDIDATE

        # -----------------------------------------------------
        # Explicitly stale.
        # -----------------------------------------------------

        if memory.status == MemoryStatus.STALE:
            return MemoryHealth.ARCHIVAL_CANDIDATE

        # -----------------------------------------------------
        # Active memories with very weak confidence.
        # -----------------------------------------------------

        if (
            confidence
            <= self.archival_confidence
        ):
            return MemoryHealth.ARCHIVAL_CANDIDATE

        if (
            confidence
            <= self.stale_confidence
        ):
            return MemoryHealth.STALE_CANDIDATE

        # -----------------------------------------------------
        # Age-based classification.
        # -----------------------------------------------------

        if (
            age_days
            >= self.archival_days
        ):
            return MemoryHealth.ARCHIVAL_CANDIDATE

        if (
            age_days
            >= self.stale_days
        ):
            return MemoryHealth.STALE_CANDIDATE

        if (
            age_days
            >= self.aging_days
        ):
            return MemoryHealth.AGING

        return MemoryHealth.HEALTHY

    # =========================================================
    # HEALTH SCORE
    # =========================================================

    def _health_score(
        self,
        memory: Memory,
        age_days: float,
        importance: float,
        confidence: float,
    ) -> float:
        """
        Calculate a continuous health score.

        Higher is healthier.

        Components:

            importance  30%
            confidence  40%
            freshness   30%
        """

        if memory.status in {
            MemoryStatus.ARCHIVED,
            MemoryStatus.SUPERSEDED,
        }:
            return 0.0

        if memory.status == MemoryStatus.STALE:
            return min(
                0.25,
                confidence * 0.5,
            )

        freshness = self._freshness_score(
            age_days
        )

        score = (
            importance * 0.30
            + confidence * 0.40
            + freshness * 0.30
        )

        return self._clamp(
            score
        )

    def _freshness_score(
        self,
        age_days: float,
    ) -> float:
        """
        Convert memory age into a freshness score.

        Fresh memories approach 1.0.

        Memories at or beyond the archival threshold
        approach 0.0.
        """

        if age_days <= 0:
            return 1.0

        if (
            age_days
            >= self.archival_days
        ):
            return 0.0

        remaining = (
            self.archival_days
            - age_days
        )

        return self._clamp(
            remaining
            / self.archival_days
        )

    # =========================================================
    # AGE
    # =========================================================

    @staticmethod
    def _age_days(
        memory: Memory,
        now: datetime,
    ) -> float:
        """
        Calculate memory age.

        updated_at is used because the memory may have been
        refreshed after creation.
        """

        timestamp = (
            memory.updated_at
            or memory.created_at
        )

        parsed = (
            MemoryHealthAnalyzer
            ._parse_datetime(timestamp)
        )

        if parsed is None:
            return 0.0

        age = (
            now - parsed
        ).total_seconds()

        return max(
            0.0,
            age / 86400.0,
        )

    # =========================================================
    # REASONS
    # =========================================================

    @staticmethod
    def _reason(
        memory: Memory,
        health: MemoryHealth,
        age_days: float,
        confidence: float,
    ) -> str:
        """
        Generate a deterministic human-readable explanation.
        """

        if (
            health
            == MemoryHealth.HEALTHY
        ):
            return (
                "The memory is active, sufficiently "
                "recent, and has adequate confidence."
            )

        if (
            health
            == MemoryHealth.AGING
        ):
            return (
                "The memory remains active but has aged "
                "past the configured aging threshold."
            )

        if (
            health
            == MemoryHealth.STALE_CANDIDATE
        ):
            if (
                confidence
                <= MemoryHealthAnalyzer.DEFAULT_STALE_CONFIDENCE
            ):
                return (
                    "The memory has low confidence and "
                    "should be considered for staleness review."
                )

            return (
                "The memory has remained active beyond the "
                "configured staleness threshold."
            )

        if (
            memory.status
            == MemoryStatus.SUPERSEDED
        ):
            return (
                "The memory has already been superseded "
                "and should not participate in normal retrieval."
            )

        if (
            memory.status
            == MemoryStatus.ARCHIVED
        ):
            return (
                "The memory is already archived and should "
                "remain outside normal retrieval."
            )

        if (
            memory.status
            == MemoryStatus.STALE
        ):
            return (
                "The memory is explicitly marked stale and "
                "is a candidate for archival review."
            )

        return (
            "The memory has reached the configured archival "
            "threshold or has critically low confidence."
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

            return (
                MemoryHealthAnalyzer
                ._ensure_utc(parsed)
            )

        except (
            TypeError,
            ValueError,
        ):
            return None

    @staticmethod
    def _ensure_utc(
        value: datetime,
    ) -> datetime:
        """
        Ensure a datetime is timezone-aware UTC.
        """

        if value.tzinfo is None:
            value = value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )

    # =========================================================
    # CLAMP
    # =========================================================

    @staticmethod
    def _clamp(
        value: float,
    ) -> float:
        """
        Clamp a value to [0.0, 1.0].
        """

        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )