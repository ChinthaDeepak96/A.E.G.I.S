"""
A.E.G.I.S. Persistent Memory Usage Store.

Stores MemoryUsage statistics separately from the Memory record.

This layer is responsible only for persistence of usage data.

It does not:
    - modify Memory objects
    - decide maintenance actions
    - perform retrieval
    - analyze memory health
"""

from __future__ import annotations

import sqlite3

from .usage import MemoryUsage


class SQLiteMemoryUsageStore:
    """
    SQLite-backed persistence for MemoryUsage.

    Usage is stored independently from the main memory table.
    """

    TABLE_NAME = "memory_usage"

    def __init__(
        self,
        database: str,
    ):
        self.database = database

        self.connection = sqlite3.connect(
            database
        )

        self.connection.row_factory = (
            sqlite3.Row
        )

        self._create_table()

    # =========================================================
    # SCHEMA
    # =========================================================

    def _create_table(self) -> None:
        """
        Create the usage table when necessary.
        """

        self.connection.execute(
            f"""
            CREATE TABLE IF NOT EXISTS
            {self.TABLE_NAME} (
                memory_id TEXT PRIMARY KEY,

                retrieval_count INTEGER NOT NULL
                    DEFAULT 0,

                access_count INTEGER NOT NULL
                    DEFAULT 0,

                last_retrieved_at TEXT
            )
            """
        )

        self.connection.commit()

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        memory_id: str,
        usage: MemoryUsage,
    ) -> MemoryUsage:
        """
        Insert or replace usage statistics.
        """

        memory_id = memory_id.strip()

        if not memory_id:
            raise ValueError(
                "memory_id cannot be empty"
            )

        if not isinstance(
            usage,
            MemoryUsage,
        ):
            raise TypeError(
                "usage must be a MemoryUsage"
            )

        self.connection.execute(
            f"""
            INSERT INTO {self.TABLE_NAME} (
                memory_id,
                retrieval_count,
                access_count,
                last_retrieved_at
            )
            VALUES (?, ?, ?, ?)

            ON CONFLICT(memory_id)
            DO UPDATE SET
                retrieval_count =
                    excluded.retrieval_count,

                access_count =
                    excluded.access_count,

                last_retrieved_at =
                    excluded.last_retrieved_at
            """,
            (
                memory_id,
                usage.retrieval_count,
                usage.access_count,
                usage.last_retrieved_at,
            ),
        )

        self.connection.commit()

        return usage

    # =========================================================
    # GET
    # =========================================================

    def get(
        self,
        memory_id: str,
    ) -> MemoryUsage | None:
        """
        Retrieve usage statistics for one memory.
        """

        memory_id = memory_id.strip()

        if not memory_id:
            return None

        row = self.connection.execute(
            f"""
            SELECT
                retrieval_count,
                access_count,
                last_retrieved_at
            FROM {self.TABLE_NAME}
            WHERE memory_id = ?
            """,
            (
                memory_id,
            ),
        ).fetchone()

        if row is None:
            return None

        return MemoryUsage(
            retrieval_count=row[
                "retrieval_count"
            ],
            access_count=row[
                "access_count"
            ],
            last_retrieved_at=row[
                "last_retrieved_at"
            ],
        )

    # =========================================================
    # GET OR CREATE
    # =========================================================

    def get_or_create(
        self,
        memory_id: str,
    ) -> MemoryUsage:
        """
        Return existing usage or create zeroed usage.
        """

        existing = self.get(
            memory_id
        )

        if existing is not None:
            return existing

        usage = MemoryUsage()

        self.save(
            memory_id,
            usage,
        )

        return usage

    # =========================================================
    # DELETE
    # =========================================================

    def delete(
        self,
        memory_id: str,
    ) -> bool:
        """
        Delete usage statistics.

        Returns True when a row was deleted.
        """

        memory_id = memory_id.strip()

        if not memory_id:
            return False

        cursor = self.connection.execute(
            f"""
            DELETE FROM {self.TABLE_NAME}
            WHERE memory_id = ?
            """,
            (
                memory_id,
            ),
        )

        self.connection.commit()

        return cursor.rowcount > 0

    # =========================================================
    # EXISTS
    # =========================================================

    def exists(
        self,
        memory_id: str,
    ) -> bool:
        """
        Check whether usage statistics exist.
        """

        memory_id = memory_id.strip()

        if not memory_id:
            return False

        row = self.connection.execute(
            f"""
            SELECT 1
            FROM {self.TABLE_NAME}
            WHERE memory_id = ?
            LIMIT 1
            """,
            (
                memory_id,
            ),
        ).fetchone()

        return row is not None

    # =========================================================
    # COUNT
    # =========================================================

    def count(self) -> int:
        """
        Return the number of stored usage records.
        """

        row = self.connection.execute(
            f"""
            SELECT COUNT(*)
            FROM {self.TABLE_NAME}
            """
        ).fetchone()

        return int(
            row[0]
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self) -> None:
        """
        Close the SQLite connection.
        """

        self.connection.close()

    def __enter__(
        self,
    ) -> "SQLiteMemoryUsageStore":
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        self.close()