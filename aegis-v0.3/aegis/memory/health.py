"""
A.E.G.I.S. Memory Health.

Provides a deterministic, read-only assessment of memory health.

Health is calculated from:

    - lifecycle status
    - age
    - importance
    - confidence
    - retrieval usage
    - access usage
    - time since last retrieval
    - staleness metadata
    - archival metadata
    - supersession metadata

This module does NOT modify memories or the database.

Actual lifecycle mutations remain the responsibility of
MemoryManager.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from .models import Memory, MemoryStatus
from .usage import MemoryUsage


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
    Immutable health report for one memory.

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

    retrieval_count: int

    access_count: int

    days_since_retrieval: float | None

    has_been_retrieved: bool

    health_score: float

    reason: str


class MemoryHealthAnalyzer:
    """
    Deterministic memory health analyzer.

    Conservative classification rules:

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

    Usage affects the continuous health score.

    Frequent and recent retrieval improves health.

    Lack of retrieval lowers health.

    Usage does not override lifecycle status.

    This class never modifies a Memory.
    """

    DEFAULT_AGING_DAYS = 30.0

    DEFAULT_STALE_DAYS = 90.0

    DEFAULT_ARCHIVAL_DAYS = 180.0

    DEFAULT_STALE_CONFIDENCE = 0.35

    DEFAULT_ARCHIVAL_CONFIDENCE = 0.20

    # ---------------------------------------------------------
    # Health score weights.
    #
    # These affect the continuous score only.
    # Classification remains deterministic.
    # ---------------------------------------------------------

    IMPORTANCE_WEIGHT = 0.25

    CONFIDENCE_WEIGHT = 0.40

    FRESHNESS_WEIGHT = 0.25

    USAGE_WEIGHT = 0.10

    # ---------------------------------------------------------
    # Usage scoring.
    # ---------------------------------------------------------

    DEFAULT_USAGE_HALF_LIFE_DAYS = 30.0

    DEFAULT_RETRIEVAL_SATURATION = 10

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
        usage_half_life_days: float = (
            DEFAULT_USAGE_HALF_LIFE_DAYS
        ),
        retrieval_saturation: int = (
            DEFAULT_RETRIEVAL_SATURATION
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

        if usage_half_life_days <= 0:
            raise ValueError(
                "usage_half_life_days must be greater than zero"
            )

        if retrieval_saturation <= 0:
            raise ValueError(
                "retrieval_saturation must be greater than zero"
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

        self.usage_half_life_days = float(
            usage_half_life_days
        )

        self.retrieval_saturation = int(
            retrieval_saturation
        )

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(
        self,
        memory: Memory,
        *,
        usage: MemoryUsage | None = None,
        now: datetime | None = None,
    ) -> MemoryHealthReport:
        """
        Analyze one memory.

        The memory and usage objects are never modified.
        """

        if not isinstance(
            memory,
            Memory,
        ):
            raise TypeError(
                "memory must be a Memory instance"
            )

        if usage is not None and not isinstance(
            usage,
            MemoryUsage,
        ):
            raise TypeError(
                "usage must be a MemoryUsage instance"
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

        usage_snapshot = (
            usage
            if usage is not None
            else MemoryUsage()
        )

        retrieval_count = max(
            0,
            int(
                usage_snapshot.retrieval_count
            ),
        )

        access_count = max(
            0,
            int(
                usage_snapshot.access_count
            ),
        )

        days_since_retrieval = (
            usage_snapshot.days_since_retrieval(
                now=current_time
            )
        )

        has_been_retrieved = (
            usage_snapshot.has_been_retrieved()
        )

        health_score = self._health_score(
            memory=memory,
            age_days=age_days,
            importance=importance,
            confidence=confidence,
            usage=usage_snapshot,
            days_since_retrieval=(
                days_since_retrieval
            ),
        )

        reason = self._reason(
            memory=memory,
            health=health,
            age_days=age_days,
            confidence=confidence,
            usage=usage_snapshot,
            days_since_retrieval=(
                days_since_retrieval
            ),
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
            retrieval_count=retrieval_count,
            access_count=access_count,
            days_since_retrieval=(
                days_since_retrieval
            ),
            has_been_retrieved=(
                has_been_retrieved
            ),
            health_score=health_score,
            reason=reason,
        )

    def analyze_many(
        self,
        memories: list[Memory],
        *,
        usages: dict[str, MemoryUsage] | None = None,
        now: datetime | None = None,
    ) -> list[MemoryHealthReport]:
        """
        Analyze multiple memories.

        Results preserve input order.

        `usages` maps memory IDs to MemoryUsage objects.
        """

        usage_map = (
            usages
            if usages is not None
            else {}
        )

        return [
            self.analyze(
                memory,
                usage=usage_map.get(
                    memory.id
                ),
                now=now,
            )
            for memory in memories
        ]

    def classify(
        self,
        memory: Memory,
        *,
        usage: MemoryUsage | None = None,
        now: datetime | None = None,
    ) -> MemoryHealth:
        """
        Return only the health classification.
        """

        return self.analyze(
            memory,
            usage=usage,
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

        if memory.status in {
            MemoryStatus.ARCHIVED,
            MemoryStatus.SUPERSEDED,
        }:
            return MemoryHealth.ARCHIVAL_CANDIDATE

        if memory.status == MemoryStatus.STALE:
            return MemoryHealth.ARCHIVAL_CANDIDATE

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
        *,
        memory: Memory,
        age_days: float,
        importance: float,
        confidence: float,
        usage: MemoryUsage,
        days_since_retrieval: float | None,
    ) -> float:
        """
        Calculate a continuous health score.

        Higher is healthier.

        Components:

            importance  25%
            confidence  35%
            freshness   20%
            usage       20%

        Lifecycle state can impose a hard ceiling.
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

        usage_score = self._usage_score(
            usage,
            days_since_retrieval,
        )

        score = (
            importance
            * self.IMPORTANCE_WEIGHT
            + confidence
            * self.CONFIDENCE_WEIGHT
            + freshness
            * self.FRESHNESS_WEIGHT
            + usage_score
            * self.USAGE_WEIGHT
        )

        return self._clamp(
            score
        )

    def _usage_score(
        self,
        usage: MemoryUsage,
        days_since_retrieval: float | None,
    ) -> float:
        """
        Calculate usage health.

        A memory that has never been retrieved receives 0.

        Retrieved memories gain a score based on:

            - retrieval frequency
            - retrieval recency
        """

        retrieval_count = max(
            0,
            int(
                usage.retrieval_count
            ),
        )

        if retrieval_count <= 0:
            return 0.0

        frequency = self._clamp(
            retrieval_count
            / self.retrieval_saturation
        )

        if days_since_retrieval is None:
            recency = 0.0

        else:
            recency = self._decay(
                days_since_retrieval,
                self.usage_half_life_days,
            )

        # -----------------------------------------------------
        # Frequency and recency both matter.
        #
        # A frequently used memory can remain useful even when
        # its latest access was not extremely recent.
        # -----------------------------------------------------

        return self._clamp(
            0.60 * frequency
            + 0.40 * recency
        )

    @staticmethod
    def _decay(
        age_days: float,
        half_life_days: float,
    ) -> float:
        """
        Exponential half-life decay.
        """

        if age_days <= 0:
            return 1.0

        if half_life_days <= 0:
            return 0.0

        import math

        return max(
            0.0,
            min(
                1.0,
                math.pow(
                    0.5,
                    age_days
                    / half_life_days,
                ),
            ),
        )

    def _freshness_score(
        self,
        age_days: float,
    ) -> float:
        """
        Convert memory age into freshness.

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
    # REASONS
    # =========================================================

    @staticmethod
    def _reason(
        *,
        memory: Memory,
        health: MemoryHealth,
        age_days: float,
        confidence: float,
        usage: MemoryUsage,
        days_since_retrieval: float | None,
    ) -> str:
        """
        Generate a deterministic human-readable explanation.
        """

        if (
            health
            == MemoryHealth.HEALTHY
        ):
            if usage.has_been_retrieved():
                return (
                    "The memory is active, sufficiently "
                    "recent, reliable, and has demonstrated "
                    "retrieval usage."
                )

            return (
                "The memory is active, sufficiently recent, "
                "and has adequate confidence."
            )

        if (
            health
            == MemoryHealth.AGING
        ):
            if usage.has_been_retrieved():
                return (
                    "The memory remains active but has aged "
                    "past the configured aging threshold; "
                    "retrieval activity indicates continued use."
                )

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

            if usage.has_been_retrieved():
                return (
                    "The memory has remained active beyond "
                    "the configured staleness threshold, "
                    "but retrieval activity indicates that "
                    "it may still be useful."
                )

            return (
                "The memory has remained active beyond the "
                "configured staleness threshold without "
                "recorded retrieval activity."
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
    # AGE
    # =========================================================

    @staticmethod
    def _age_days(
        memory: Memory,
        now: datetime,
    ) -> float:
        """
        Calculate memory age.

        updated_at is preferred because a memory may have been
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