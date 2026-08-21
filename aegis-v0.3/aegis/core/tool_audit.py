"""
A.E.G.I.S. Tool Execution Audit Records.

This module defines the immutable evidence record produced by the
tool-execution layer.

The audit record does not execute tools and does not decide whether
a tool is safe. Those responsibilities belong to ToolGateway and
Guardian respectively.

Its purpose is to preserve what happened during a tool execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class ToolAuditRecord:
    """
    Evidence record describing one tool execution attempt.

    Fields:

        tool_name:
            Name of the tool involved.

        risk_category:
            Guardian risk classification.

        decision:
            Guardian decision, such as ALLOW or CONFIRM.

        confirmed:
            Whether explicit confirmation was provided.

        executed:
            Whether the handler was actually invoked.

        success:
            Whether the handler execution succeeded.

        arguments:
            Arguments supplied to the tool handler.

        result:
            Successful handler result, when available.

        error:
            Error information when execution failed.

        timestamp:
            UTC timestamp representing when the record was created.
    """

    tool_name: str

    risk_category: str

    decision: str

    confirmed: bool

    executed: bool

    success: bool

    arguments: dict[str, Any]

    result: str | None = None

    error: str | None = None

    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        """
        Normalize the audit record after construction.
        """

        if self.timestamp is None:
            self.timestamp = datetime.now(
                timezone.utc
            )
        else:
            self.timestamp = (
                self._ensure_utc(
                    self.timestamp
                )
            )

        # Make a shallow copy so the record does not
        # share the caller's mutable dictionary.
        self.arguments = dict(
            self.arguments
        )

    # =========================================================
    # SERIALIZATION
    # =========================================================

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the audit record into a dictionary.
        """

        return {
            "tool_name": self.tool_name,
            "risk_category": self.risk_category,
            "decision": self.decision,
            "confirmed": self.confirmed,
            "executed": self.executed,
            "success": self.success,
            "arguments": dict(
                self.arguments
            ),
            "result": self.result,
            "error": self.error,
            "timestamp": (
                self.timestamp.isoformat()
                if self.timestamp is not None
                else None
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any] | None,
    ) -> "ToolAuditRecord":
        """
        Reconstruct an audit record from a dictionary.

        Missing optional values are handled safely so older
        audit records remain readable.
        """

        if not data:
            raise ValueError(
                "Audit record data cannot be empty."
            )

        timestamp_value = data.get(
            "timestamp"
        )

        timestamp = (
            cls._parse_datetime(
                timestamp_value
            )
            if timestamp_value is not None
            else None
        )

        arguments = data.get(
            "arguments",
            {},
        )

        if not isinstance(
            arguments,
            dict,
        ):
            arguments = {}

        return cls(
            tool_name=str(
                data.get(
                    "tool_name",
                    "",
                )
            ),
            risk_category=str(
                data.get(
                    "risk_category",
                    "",
                )
            ),
            decision=str(
                data.get(
                    "decision",
                    "",
                )
            ),
            confirmed=bool(
                data.get(
                    "confirmed",
                    False,
                )
            ),
            executed=bool(
                data.get(
                    "executed",
                    False,
                )
            ),
            success=bool(
                data.get(
                    "success",
                    False,
                )
            ),
            arguments=dict(
                arguments
            ),
            result=(
                str(
                    data["result"]
                )
                if data.get(
                    "result"
                ) is not None
                else None
            ),
            error=(
                str(
                    data["error"]
                )
                if data.get(
                    "error"
                ) is not None
                else None
            ),
            timestamp=timestamp,
        )

    # =========================================================
    # DATETIME HELPERS
    # =========================================================

    @staticmethod
    def _parse_datetime(
        value: Any,
    ) -> datetime | None:
        """
        Parse an ISO-formatted timestamp.
        """

        if isinstance(
            value,
            datetime,
        ):
            return ToolAuditRecord._ensure_utc(
                value
            )

        if not isinstance(
            value,
            str,
        ):
            return None

        try:
            parsed = datetime.fromisoformat(
                value
            )

            return ToolAuditRecord._ensure_utc(
                parsed
            )

        except ValueError:
            return None

    @staticmethod
    def _ensure_utc(
        value: datetime,
    ) -> datetime:
        """
        Normalize a datetime to timezone-aware UTC.
        """

        if value.tzinfo is None:
            value = value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )