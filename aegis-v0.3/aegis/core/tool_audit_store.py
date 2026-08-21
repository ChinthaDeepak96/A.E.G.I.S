"""
A.E.G.I.S. persistent tool audit storage.

Tool audit records are operational evidence and are intentionally
stored separately from conversational memory.

The store uses SQLite so audit history survives process restarts
without introducing another database dependency.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.tool_audit import ToolAuditRecord
from core.tool_audit_analyzer import (
    ToolAuditAnalyzer,
)

from core.tool_audit_security import (
    ToolAuditSecurityAnalyzer,
)

class SQLiteToolAuditStore:
    """
    SQLite-backed storage for ToolAuditRecord objects.
    """

    def __init__(
        self,
        database_path: str = "data/aegis_tool_audit.db",
    ):
        self.database_path = str(
            database_path
        )

        path = Path(
            self.database_path
        )

        # Ensure the parent directory exists when
        # a directory component is supplied.
        if path.parent:
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    # =========================================================
    # DATABASE
    # =========================================================

    def _connect(
        self,
    ) -> sqlite3.Connection:
        """
        Open a SQLite connection.
        """

        connection = sqlite3.connect(
            self.database_path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        return connection

    def _initialize(self) -> None:
        """
        Create the audit table when necessary.
        """

        connection = self._connect()

        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_audit (
                    id TEXT PRIMARY KEY,
                    tool_name TEXT NOT NULL,
                    risk_category TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    confirmed INTEGER NOT NULL,
                    executed INTEGER NOT NULL,
                    success INTEGER NOT NULL,
                    arguments TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    timestamp TEXT NOT NULL
                )
                """
            )

            connection.commit()

        finally:
            connection.close()

    def close(self) -> None:
        """
        Close the store.

        Connections are intentionally short-lived, so there is no
        persistent connection to close. The method exists as a
        lifecycle-friendly public API.
        """

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        record: ToolAuditRecord,
    ) -> ToolAuditRecord:
        """
        Persist an audit record.

        A UUID is assigned to the record when it does not already
        have an ID.
        """

        if not isinstance(
            record,
            ToolAuditRecord,
        ):
            raise TypeError(
                "record must be a ToolAuditRecord"
            )

        record_id = getattr(
            record,
            "id",
            None,
        )

        if not record_id:
            record_id = str(
                uuid.uuid4()
            )
            record.id = record_id

        if record.timestamp is None:
            record.timestamp = datetime.now(
                timezone.utc
            )

        connection = self._connect()

        try:
            connection.execute(
                """
                INSERT OR REPLACE INTO tool_audit (
                    id,
                    tool_name,
                    risk_category,
                    decision,
                    confirmed,
                    executed,
                    success,
                    arguments,
                    result,
                    error,
                    timestamp
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    record.id,
                    record.tool_name,
                    record.risk_category,
                    record.decision,
                    int(record.confirmed),
                    int(record.executed),
                    int(record.success),
                    json.dumps(
                        record.arguments,
                        ensure_ascii=False,
                    ),
                    record.result,
                    record.error,
                    record.timestamp.isoformat(),
                ),
            )

            connection.commit()

        finally:
            connection.close()

        return record

    # =========================================================
    # GET
    # =========================================================

    def get(
        self,
        record_id: str,
    ) -> ToolAuditRecord | None:
        """
        Retrieve one audit record by ID.
        """

        if not record_id:
            return None

        record_id = record_id.strip()

        if not record_id:
            return None

        connection = self._connect()

        try:
            row = connection.execute(
                """
                SELECT *
                FROM tool_audit
                WHERE id = ?
                """,
                (
                    record_id,
                ),
            ).fetchone()

        finally:
            connection.close()

        if row is None:
            return None

        return self._row_to_record(
            row
        )

    # =========================================================
    # LIST
    # =========================================================

    def list(
        self,
    ) -> list[ToolAuditRecord]:
        """
        Return all audit records, newest first.
        """

        connection = self._connect()

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM tool_audit
                ORDER BY timestamp DESC
                """
            ).fetchall()

        finally:
            connection.close()

        return [
            self._row_to_record(row)
            for row in rows
        ]

    # =========================================================
    # RECENT
    # =========================================================

    def recent(
        self,
        limit: int = 20,
    ) -> list[ToolAuditRecord]:
        """
        Return the most recent audit records.
        """

        if limit <= 0:
            return []

        connection = self._connect()

        try:
            rows = connection.execute(
                """
                SELECT *
                FROM tool_audit
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (
                    int(limit),
                ),
            ).fetchall()

        finally:
            connection.close()

        return [
            self._row_to_record(row)
            for row in rows
        ]
    
    # =========================================================
    # ANALYSIS
    # =========================================================

    def analyze(
        self,
    ) -> ToolAuditAnalyzer:
        """
        Create a read-only analyzer over the current audit history.

        The analyzer receives a snapshot of the persisted records.
        It does not modify the database.
        """

        return ToolAuditAnalyzer(
            self.list()
        )

    # =========================================================
    # SECURITY ANALYSIS
    # =========================================================

    def security_analysis(
        self,
    ) -> ToolAuditSecurityAnalyzer:
        """
        Create a read-only security analyzer over the
        current persisted audit history.

        The analyzer receives the current audit snapshot
        and does not modify the database.
        """

        return ToolAuditSecurityAnalyzer(
            self.analyze()
        )
    
    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self) -> int:
        """
        Delete all audit records.

        Returns the number of deleted records.
        """

        connection = self._connect()

        try:
            cursor = connection.execute(
                """
                DELETE FROM tool_audit
                """
            )

            connection.commit()

            return cursor.rowcount

        finally:
            connection.close()

    # =========================================================
    # ROW CONVERSION
    # =========================================================

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
    ) -> ToolAuditRecord:
        """
        Convert a SQLite row into ToolAuditRecord.
        """

        arguments: dict[str, Any]

        try:
            decoded = json.loads(
                row["arguments"]
            )

            if isinstance(
                decoded,
                dict,
            ):
                arguments = decoded
            else:
                arguments = {}

        except (
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            arguments = {}

        timestamp = (
            SQLiteToolAuditStore
            ._parse_datetime(
                row["timestamp"]
            )
        )

        record = ToolAuditRecord(
            tool_name=row["tool_name"],
            risk_category=(
                row["risk_category"]
            ),
            decision=row["decision"],
            confirmed=bool(
                row["confirmed"]
            ),
            executed=bool(
                row["executed"]
            ),
            success=bool(
                row["success"]
            ),
            arguments=arguments,
            result=row["result"],
            error=row["error"],
            timestamp=timestamp,
        )

        record.id = row["id"]

        return record

    @staticmethod
    def _parse_datetime(
        value: str | None,
    ) -> datetime:
        """
        Parse an ISO timestamp and normalize it to UTC.
        """

        if not value:
            return datetime.now(
                timezone.utc
            )

        try:
            parsed = datetime.fromisoformat(
                value
            )

        except ValueError:
            return datetime.now(
                timezone.utc
            )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )