"""
A.E.G.I.S. Memory Usage Tracking.

Tracks how frequently a memory is accessed or retrieved.

This module provides usage metadata independently from the
Memory lifecycle and storage layers.

It does not decide whether a memory should be archived,
staled, deleted, or modified.

Those decisions remain the responsibility of the health,
maintenance, and Guardian layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass
class MemoryUsage:
    """
    Mutable usage statistics for a memory.

    Fields:

        retrieval_count:
            Number of successful retrieval/access events.

        last_retrieved_at:
            Timestamp of the most recent retrieval.

        access_count:
            Total number of explicit accesses.

    Usage statistics are intentionally separate from Memory so
    the lifecycle model does not become overloaded with behavior.
    """

    retrieval_count: int = 0

    last_retrieved_at: str | None = None

    access_count: int = 0

    def record_retrieval(
        self,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Record one retrieval event.
        """

        self.retrieval_count += 1

        self.access_count += 1

        now = (
            timestamp
            if timestamp is not None
            else datetime.now(
                timezone.utc
            )
        )

        self.last_retrieved_at = (
            self._format_datetime(now)
        )

    def record_access(
        self,
        *,
        timestamp: datetime | None = None,
    ) -> None:
        """
        Record one general access event.

        Unlike retrieval, this does not increment
        retrieval_count.
        """

        self.access_count += 1

    def days_since_retrieval(
        self,
        *,
        now: datetime | None = None,
    ) -> float | None:
        """
        Return the number of days since the last retrieval.

        Returns None when the memory has never been retrieved.
        """

        if not self.last_retrieved_at:
            return None

        retrieved_at = (
            self._parse_datetime(
                self.last_retrieved_at
            )
        )

        if retrieved_at is None:
            return None

        current = (
            now
            if now is not None
            else datetime.now(
                timezone.utc
            )
        )

        current = self._ensure_utc(
            current
        )

        elapsed = (
            current - retrieved_at
        ).total_seconds()

        return max(
            0.0,
            elapsed / 86400.0,
        )

    def has_been_retrieved(self) -> bool:
        """
        Return True when at least one retrieval occurred.
        """

        return self.retrieval_count > 0

    def reset(self) -> None:
        """
        Reset usage statistics.
        """

        self.retrieval_count = 0

        self.last_retrieved_at = None

        self.access_count = 0

    def to_dict(self) -> dict[str, object]:
        """
        Serialize usage statistics.
        """

        return {
            "retrieval_count": self.retrieval_count,
            "last_retrieved_at": (
                self.last_retrieved_at
            ),
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object] | None,
    ) -> "MemoryUsage":
        """
        Reconstruct usage statistics from a dictionary.
        """

        if not data:
            return cls()

        retrieval_count = int(
            data.get(
                "retrieval_count",
                0,
            )
        )

        access_count = int(
            data.get(
                "access_count",
                0,
            )
        )

        last_retrieved_at = data.get(
            "last_retrieved_at"
        )

        if last_retrieved_at is not None:
            last_retrieved_at = str(
                last_retrieved_at
            )

        return cls(
            retrieval_count=max(
                0,
                retrieval_count,
            ),
            last_retrieved_at=(
                last_retrieved_at
            ),
            access_count=max(
                0,
                access_count,
            ),
        )

    # =========================================================
    # DATETIME HELPERS
    # =========================================================

    @staticmethod
    def _format_datetime(
        value: datetime,
    ) -> str:
        """
        Convert datetime to normalized UTC ISO format.
        """

        return (
            MemoryUsage
            ._ensure_utc(value)
            .isoformat()
        )

    @staticmethod
    def _parse_datetime(
        value: str | None,
    ) -> datetime | None:
        """
        Parse an ISO timestamp.
        """

        if not value:
            return None

        try:
            parsed = datetime.fromisoformat(
                value
            )

            return (
                MemoryUsage
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