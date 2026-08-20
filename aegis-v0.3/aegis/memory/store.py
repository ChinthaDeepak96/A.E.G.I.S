from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import Memory, MemoryType


SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL,
    confidence REAL NOT NULL,
    sensitivity REAL NOT NULL,
    source TEXT NOT NULL,
    tags_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_memories_type
    ON memories(memory_type);

CREATE INDEX IF NOT EXISTS idx_memories_created
    ON memories(created_at);

CREATE INDEX IF NOT EXISTS idx_memories_importance
    ON memories(importance);
"""


class SQLiteMemoryStore:
    """
    SQLite-backed persistent memory store.

    Supports both:

        data/aegis_memory.db

    and:

        :memory:

    The in-memory database keeps one persistent connection so all
    operations use the same SQLite database.
    """

    def __init__(
        self,
        path: str | Path = "data/aegis_memory.db",
    ):
        self.path = Path(path)

        self._connection: sqlite3.Connection | None = None

        # ---------------------------------------------------------
        # Special handling for SQLite in-memory databases.
        #
        # Each sqlite3.connect(":memory:") normally creates a
        # completely separate database. Therefore we must keep
        # one connection alive for the lifetime of this store.
        # ---------------------------------------------------------

        if str(path) == ":memory:":
            self._connection = sqlite3.connect(
                ":memory:"
            )

            self._connection.row_factory = sqlite3.Row

            self._initialize()

        else:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._initialize()

    # =========================================================
    # CONNECTION MANAGEMENT
    # =========================================================

    def _connect(self) -> sqlite3.Connection:
        """
        Return the appropriate SQLite connection.

        Persistent database:
            Creates a normal short-lived connection.

        In-memory database:
            Returns the single persistent connection.
        """

        if self._connection is not None:
            return self._connection

        connection = sqlite3.connect(
            self.path
        )

        connection.row_factory = sqlite3.Row

        return connection

    def _initialize(self) -> None:
        """
        Create the database schema if it doesn't already exist.
        """

        conn = self._connect()

        if self._connection is not None:
            conn.executescript(SCHEMA)
            conn.commit()
            return

        with conn:
            conn.executescript(SCHEMA)

        conn.close()

    def close(self) -> None:
        """
        Close the persistent in-memory connection.

        Normal file-backed connections are opened and closed per
        operation, so there is nothing to close here.
        """

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # =========================================================
    # SAVE
    # =========================================================

    def save(
        self,
        memory: Memory,
    ) -> Memory:
        conn = self._connect()

        if self._connection is not None:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (
                    id,
                    memory_type,
                    content,
                    importance,
                    confidence,
                    sensitivity,
                    source,
                    tags_json,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                memory.to_record(),
            )

            conn.commit()

            return memory

        with conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (
                    id,
                    memory_type,
                    content,
                    importance,
                    confidence,
                    sensitivity,
                    source,
                    tags_json,
                    metadata_json,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                memory.to_record(),
            )

        conn.close()

        return memory

    # =========================================================
    # GET
    # =========================================================

    def get(
        self,
        memory_id: str,
    ) -> Memory | None:

        conn = self._connect()

        row = conn.execute(
            """
            SELECT
                id,
                memory_type,
                content,
                importance,
                confidence,
                sensitivity,
                source,
                tags_json,
                metadata_json,
                created_at,
                updated_at
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        if self._connection is None:
            conn.close()

        return (
            Memory.from_row(tuple(row))
            if row
            else None
        )

    # =========================================================
    # DELETE
    # =========================================================

    def delete(
        self,
        memory_id: str,
    ) -> bool:

        conn = self._connect()

        cursor = conn.execute(
            """
            DELETE FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        )

        if self._connection is not None:
            conn.commit()

        else:
            conn.commit()
            conn.close()

        return cursor.rowcount > 0

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self) -> int:

        conn = self._connect()

        cursor = conn.execute(
            "DELETE FROM memories"
        )

        if self._connection is not None:
            conn.commit()

        else:
            conn.commit()
            conn.close()

        return cursor.rowcount

    # =========================================================
    # LIST
    # =========================================================

    def list(
        self,
        memory_type: MemoryType | None = None,
        limit: int = 100,
    ) -> list[Memory]:

        query = """
        SELECT
            id,
            memory_type,
            content,
            importance,
            confidence,
            sensitivity,
            source,
            tags_json,
            metadata_json,
            created_at,
            updated_at
        FROM memories
        """

        params: list[object] = []

        if memory_type:
            query += """
            WHERE memory_type = ?
            """

            params.append(
                memory_type.value
            )

        query += """
        ORDER BY created_at DESC
        LIMIT ?
        """

        params.append(
            max(1, int(limit))
        )

        conn = self._connect()

        rows = conn.execute(
            query,
            params,
        ).fetchall()

        if self._connection is None:
            conn.close()

        return [
            Memory.from_row(tuple(row))
            for row in rows
        ]

    # =========================================================
    # TEXT SEARCH
    # =========================================================

    def search_text(
        self,
        query: str,
        limit: int = 10,
    ) -> list[Memory]:

        terms = [
            term.strip().lower()
            for term in query.split()
            if term.strip()
        ]

        if not terms:
            return []

        conn = self._connect()

        rows = conn.execute(
            """
            SELECT
                id,
                memory_type,
                content,
                importance,
                confidence,
                sensitivity,
                source,
                tags_json,
                metadata_json,
                created_at,
                updated_at
            FROM memories
            """
        ).fetchall()

        if self._connection is None:
            conn.close()

        memories = [
            Memory.from_row(tuple(row))
            for row in rows
        ]

        def score(
            memory: Memory,
        ) -> float:

            text = (
                memory.content
                + " "
                + " ".join(memory.tags)
            ).lower()

            hits = sum(
                1
                for term in terms
                if term in text
            )

            return (
                hits * 2.0
                + memory.importance
                + memory.confidence
            )

        ranked = sorted(
            (
                memory
                for memory in memories
                if score(memory) > 0
            ),
            key=score,
            reverse=True,
        )

        return ranked[
            : max(1, int(limit))
        ]