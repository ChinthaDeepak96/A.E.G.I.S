from __future__ import annotations

import sqlite3
from pathlib import Path

from .models import Memory, MemoryStatus, MemoryType


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
    status TEXT NOT NULL DEFAULT 'active',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    stale_at TEXT,
    archived_at TEXT,
    superseded_by TEXT
);
"""


class SQLiteMemoryStore:
    """
    SQLite-backed persistent memory store.

    Supports:
        - normal file-backed SQLite databases
        - SQLite :memory: databases

    Also performs automatic migration of older A.E.G.I.S.
    memory databases to the v0.4.3 lifecycle schema.
    """

    def __init__(
        self,
        path: str | Path = "data/aegis_memory.db",
    ):
        self.path = Path(path)

        self._connection: sqlite3.Connection | None = None

        # ---------------------------------------------------------
        # In-memory database
        # ---------------------------------------------------------

        if str(path) == ":memory:":
            self._connection = sqlite3.connect(
                ":memory:"
            )

            self._connection.row_factory = sqlite3.Row

        # ---------------------------------------------------------
        # File-backed database
        # ---------------------------------------------------------

        else:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    # =========================================================
    # CONNECTION
    # =========================================================

    def _connect(self) -> sqlite3.Connection:
        """
        Return the appropriate SQLite connection.
        """

        if self._connection is not None:
            return self._connection

        connection = sqlite3.connect(
            self.path
        )

        connection.row_factory = sqlite3.Row

        return connection

    # =========================================================
    # INITIALIZATION + MIGRATION
    # =========================================================

    def _initialize(self) -> None:
        """
        Create the database or migrate an existing database.
        """

        conn = self._connect()

        # Check whether the memories table already exists.
        table_exists = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'memories'
            """
        ).fetchone()

        # ---------------------------------------------------------
        # Existing database
        # ---------------------------------------------------------

        if table_exists:

            self._migrate_schema(
                conn
            )

        # ---------------------------------------------------------
        # New database
        # ---------------------------------------------------------

        else:

            conn.executescript(
                SCHEMA
            )

        # ---------------------------------------------------------
        # Indexes
        #
        # These are deliberately created AFTER migration so an old
        # database does not fail because status doesn't exist yet.
        # ---------------------------------------------------------

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_type
            ON memories(memory_type)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_created
            ON memories(created_at)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_importance
            ON memories(importance)
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_status
            ON memories(status)
            """
        )

        conn.commit()

        if self._connection is None:
            conn.close()

    def _migrate_schema(
        self,
        conn: sqlite3.Connection,
    ) -> None:
        """
        Upgrade a pre-v0.4.3 A.E.G.I.S. memory database.

        Existing memories receive:
            status = active

        New lifecycle columns are added only when missing.
        """

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(memories)"
            ).fetchall()
        }

        # ---------------------------------------------------------
        # Lifecycle columns
        # ---------------------------------------------------------

        if "status" not in columns:

            conn.execute(
                """
                ALTER TABLE memories
                ADD COLUMN status TEXT
                NOT NULL
                DEFAULT 'active'
                """
            )

        if "stale_at" not in columns:

            conn.execute(
                """
                ALTER TABLE memories
                ADD COLUMN stale_at TEXT
                """
            )

        if "archived_at" not in columns:

            conn.execute(
                """
                ALTER TABLE memories
                ADD COLUMN archived_at TEXT
                """
            )

        if "superseded_by" not in columns:

            conn.execute(
                """
                ALTER TABLE memories
                ADD COLUMN superseded_by TEXT
                """
            )

        # ---------------------------------------------------------
        # Defensive migration
        #
        # Existing rows should always be considered ACTIVE.
        # ---------------------------------------------------------

        conn.execute(
            """
            UPDATE memories
            SET status = 'active'
            WHERE status IS NULL
               OR status = ''
            """
        )

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self) -> None:
        """
        Close the persistent in-memory connection.
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
                status,
                created_at,
                updated_at,
                stale_at,
                archived_at,
                superseded_by
            )
            VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?
            )
            """,
            memory.to_record(),
        )

        conn.commit()

        if self._connection is None:
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
                status,
                created_at,
                updated_at,
                stale_at,
                archived_at,
                superseded_by
            FROM memories
            WHERE id = ?
            """,
            (memory_id,),
        ).fetchone()

        if self._connection is None:
            conn.close()

        if row is None:
            return None

        return Memory.from_row(
            tuple(row)
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

        conn.commit()

        if self._connection is None:
            conn.close()

        return cursor.rowcount > 0

    # =========================================================
    # CLEAR
    # =========================================================

    def clear(self) -> int:

        conn = self._connect()

        cursor = conn.execute(
            """
            DELETE FROM memories
            """
        )

        conn.commit()

        if self._connection is None:
            conn.close()

        return cursor.rowcount

    # =========================================================
    # LIST
    # =========================================================

    def list(
        self,
        memory_type: MemoryType | None = None,
        status: MemoryStatus | None = None,
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
            status,
            created_at,
            updated_at,
            stale_at,
            archived_at,
            superseded_by
        FROM memories
        """

        conditions: list[str] = []
        params: list[object] = []

        if memory_type is not None:

            conditions.append(
                "memory_type = ?"
            )

            params.append(
                memory_type.value
            )

        if status is not None:

            conditions.append(
                "status = ?"
            )

            params.append(
                status.value
            )

        if conditions:

            query += (
                " WHERE "
                + " AND ".join(conditions)
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
            Memory.from_row(
                tuple(row)
            )
            for row in rows
        ]

    # =========================================================
    # UPDATE STATUS
    # =========================================================

    def update_status(
        self,
        memory_id: str,
        status: MemoryStatus,
        *,
        superseded_by: str | None = None,
    ) -> Memory | None:

        memory = self.get(
            memory_id
        )

        if memory is None:
            return None

        if status == MemoryStatus.NEW:

            memory.status = (
                MemoryStatus.NEW
            )

            memory.touch()

        elif status == MemoryStatus.ACTIVE:

            memory.activate()

        elif status == MemoryStatus.STALE:

            memory.mark_stale()

        elif status == MemoryStatus.ARCHIVED:

            memory.archive()

        elif status == MemoryStatus.SUPERSEDED:

            if not superseded_by:
                raise ValueError(
                    "superseded_by is required "
                    "for SUPERSEDED memories"
                )

            memory.supersede(
                superseded_by
            )

        return self.save(
            memory
        )

        # =========================================================
    # TEXT SEARCH
    # =========================================================

    def search_text(
        self,
        query: str,
        limit: int = 10,
        include_inactive: bool = False,
    ) -> list[Memory]:
        """
        Perform deterministic lexical memory search.

        Ranking signals:

            1. Exact phrase match
            2. Exact content-token matches
            3. Exact tag-token matches
            4. Partial lexical matches
            5. Query coverage
            6. Importance
            7. Confidence

        By default, only ACTIVE memories participate.

        This layer performs candidate discovery. The higher-level
        MemoryRetriever may apply additional ranking afterward.
        """

        import re

        query = query.strip()

        if not query:
            return []

        limit = max(
            1,
            int(limit),
        )

        normalized_query = query.lower()

        # -----------------------------------------------------
        # Tokenize query
        # -----------------------------------------------------

        def tokenize(text: str) -> list[str]:
            return re.findall(
                r"\b[a-z0-9_]+\b",
                text.lower(),
            )

        query_tokens = {
            token
            for token in tokenize(
                normalized_query
            )
            if len(token) >= 2
        }

        if not query_tokens:
            return []

        # -----------------------------------------------------
        # Load eligible memories
        # -----------------------------------------------------

        conn = self._connect()

        sql = """
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
            status,
            created_at,
            updated_at,
            stale_at,
            archived_at,
            superseded_by
        FROM memories
        """

        params: list[object] = []

        if not include_inactive:
            sql += """
            WHERE status = ?
            """

            params.append(
                MemoryStatus.ACTIVE.value
            )

        rows = conn.execute(
            sql,
            params,
        ).fetchall()

        if self._connection is None:
            conn.close()

        memories = [
            Memory.from_row(
                tuple(row)
            )
            for row in rows
        ]

        # -----------------------------------------------------
        # Score candidate
        # -----------------------------------------------------

        def score(memory: Memory) -> float:
            content = memory.content.lower()
            tags_text = " ".join(
                memory.tags
            ).lower()

            content_tokens = set(
                tokenize(content)
            )

            tag_tokens = set(
                tokenize(tags_text)
            )

            exact_content_hits = (
                query_tokens
                & content_tokens
            )

            exact_tag_hits = (
                query_tokens
                & tag_tokens
            )

            score_value = 0.0

            # Exact complete phrase.
            if normalized_query in content:
                score_value += 12.0

            # Exact token matches.
            score_value += (
                len(exact_content_hits)
                * 4.0
            )

            score_value += (
                len(exact_tag_hits)
                * 3.0
            )

            # Partial lexical matches.
            for term in query_tokens:
                if term in content:
                    score_value += 1.0

                if term in tags_text:
                    score_value += 0.5

            # Query coverage.
            matched_tokens = (
                exact_content_hits
                | exact_tag_hits
            )

            coverage = (
                len(matched_tokens)
                / len(query_tokens)
            )

            score_value += (
                coverage * 4.0
            )

            # Memory quality signals.
            score_value += (
                memory.importance * 1.5
            )

            score_value += (
                memory.confidence
            )

            return score_value

        # -----------------------------------------------------
        # Rank
        # -----------------------------------------------------

        ranked = sorted(
            (
                memory
                for memory in memories
                if score(memory) > 0
            ),
            key=lambda memory: (
                score(memory),
                memory.importance,
                memory.confidence,
                memory.updated_at,
            ),
            reverse=True,
        )

        return ranked[:limit]