"""
A.E.G.I.S. SQLite Memory Store.

Responsibilities:

    - Persistent memory storage
    - Memory lifecycle persistence
    - Schema migration
    - Lexical candidate discovery
    - Semantic embedding persistence

The store does not generate embeddings and does not perform
semantic similarity. Those responsibilities belong to the
embedding and retrieval layers.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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

CREATE TABLE IF NOT EXISTS memory_embeddings (
    memory_id TEXT PRIMARY KEY,
    embedding BLOB NOT NULL,
    model_name TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (memory_id)
        REFERENCES memories(id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_memories_type
ON memories(memory_type);

CREATE INDEX IF NOT EXISTS idx_memories_created
ON memories(created_at);

CREATE INDEX IF NOT EXISTS idx_memories_importance
ON memories(importance);

CREATE INDEX IF NOT EXISTS idx_memories_status
ON memories(status);

CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model
ON memory_embeddings(model_name);
"""


class SQLiteMemoryStore:
    """
    SQLite-backed persistent memory store.

    Supports:

        - file-backed SQLite databases
        - SQLite :memory: databases
        - lifecycle-aware memories
        - migration of older memory databases
        - semantic embedding persistence
    """

    def __init__(
        self,
        path: str | Path = "data/aegis_memory.db",
    ):
        self.path = Path(path)

        self._connection: sqlite3.Connection | None = None

        # -----------------------------------------------------
        # In-memory database
        # -----------------------------------------------------

        if str(path) == ":memory:":
            self._connection = sqlite3.connect(
                ":memory:"
            )

            self._connection.row_factory = (
                sqlite3.Row
            )

            self._enable_foreign_keys(
                self._connection
            )

        # -----------------------------------------------------
        # File-backed database
        # -----------------------------------------------------

        else:
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._initialize()

    # =========================================================
    # CONNECTION
    # =========================================================

    @staticmethod
    def _enable_foreign_keys(
        connection: sqlite3.Connection,
    ) -> None:
        """
        Enable SQLite foreign-key enforcement.
        """

        connection.execute(
            "PRAGMA foreign_keys = ON"
        )

    def _connect(self) -> sqlite3.Connection:
        """
        Return the appropriate SQLite connection.
        """

        if self._connection is not None:
            return self._connection

        connection = sqlite3.connect(
            self.path
        )

        connection.row_factory = (
            sqlite3.Row
        )

        self._enable_foreign_keys(
            connection
        )

        return connection

    # =========================================================
    # INITIALIZATION + MIGRATION
    # =========================================================

    def _initialize(self) -> None:
        """
        Create the database or migrate an existing database.
        """

        conn = self._connect()

        table_exists = conn.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'memories'
            """
        ).fetchone()

        if table_exists:
            self._migrate_schema(conn)
        else:
            conn.executescript(
                SCHEMA
            )

        # -----------------------------------------------------
        # Ensure embedding table exists even for migrated DBs.
        # -----------------------------------------------------

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS memory_embeddings (
                memory_id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (memory_id)
                    REFERENCES memories(id)
                    ON DELETE CASCADE
            )
            """
        )

        # -----------------------------------------------------
        # Indexes
        # -----------------------------------------------------

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

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memory_embeddings_model
            ON memory_embeddings(model_name)
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
        Upgrade an older A.E.G.I.S. memory database.

        Existing memories are treated as ACTIVE unless an
        existing lifecycle value says otherwise.
        """

        columns = {
            row["name"]
            for row in conn.execute(
                "PRAGMA table_info(memories)"
            ).fetchall()
        }

        # -----------------------------------------------------
        # Lifecycle columns
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Defensive migration
        # -----------------------------------------------------

        conn.execute(
            """
            UPDATE memories
            SET status = 'active'
            WHERE status IS NULL
               OR status = ''
            """
        )

        conn.commit()

    # =========================================================
    # CLOSE
    # =========================================================

    def close(self) -> None:
        """
        Close a persistent in-memory connection.
        """

        if self._connection is not None:
            self._connection.close()
            self._connection = None

    # =========================================================
    # SAVE MEMORY
    # =========================================================

    def save(
        self,
        memory: Memory,
    ) -> Memory:
        """
        Persist a memory.
        """

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
        """
        Retrieve a memory by ID.
        """

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
        """
        Permanently delete a memory.

        Its embedding is automatically removed through the
        memory_embeddings foreign-key cascade.
        """

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
        """
        Permanently delete all memories.

        Associated embeddings are removed through the
        foreign-key cascade.
        """

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
        """
        Return recent memories with optional filters.
        """

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
                + " AND ".join(
                    conditions
                )
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
        """
        Update the lifecycle status of a memory.
        """

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
    # EMBEDDING STORAGE
    # =========================================================

    def save_embedding(
        self,
        memory_id: str,
        embedding: np.ndarray,
        model_name: str,
    ) -> None:
        """
        Persist an embedding for a memory.

        Saving another embedding for the same memory replaces
        the existing vector.
        """

        if (
            not isinstance(
                memory_id,
                str,
            )
            or not memory_id.strip()
        ):
            raise ValueError(
                "memory_id must be a non-empty string"
            )

        if (
            not isinstance(
                model_name,
                str,
            )
            or not model_name.strip()
        ):
            raise ValueError(
                "model_name must be a non-empty string"
            )

        vector = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if vector.ndim != 1:
            raise ValueError(
                "embedding must be a one-dimensional vector"
            )

        if vector.size == 0:
            raise ValueError(
                "embedding cannot be empty"
            )

        if not np.all(
            np.isfinite(vector)
        ):
            raise ValueError(
                "embedding must contain only finite values"
            )

        # Verify that the memory exists.
        memory_exists = self.get(
            memory_id
        )

        if memory_exists is None:
            raise ValueError(
                f"Cannot store embedding for unknown "
                f"memory: {memory_id}"
            )

        created_at = datetime.now(
            timezone.utc
        ).isoformat()

        conn = self._connect()

        conn.execute(
            """
            INSERT OR REPLACE INTO memory_embeddings
            (
                memory_id,
                embedding,
                model_name,
                dimensions,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                memory_id,
                sqlite3.Binary(
                    vector.tobytes()
                ),
                model_name,
                int(vector.size),
                created_at,
            ),
        )

        conn.commit()

        if self._connection is None:
            conn.close()

    def get_embedding(
        self,
        memory_id: str,
    ) -> tuple[
        np.ndarray,
        str,
        int,
        str,
    ] | None:
        """
        Retrieve an embedding.

        Returns:

            (
                embedding,
                model_name,
                dimensions,
                created_at,
            )

        or None when no embedding exists.
        """

        conn = self._connect()

        row = conn.execute(
            """
            SELECT
                embedding,
                model_name,
                dimensions,
                created_at
            FROM memory_embeddings
            WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()

        if self._connection is None:
            conn.close()

        if row is None:
            return None

        vector = np.frombuffer(
            row["embedding"],
            dtype=np.float32,
        ).copy()

        dimensions = int(
            row["dimensions"]
        )

        if vector.size != dimensions:
            raise ValueError(
                "Stored embedding dimensions do not "
                "match recorded dimensions"
            )

        return (
            vector,
            row["model_name"],
            dimensions,
            row["created_at"],
        )

    def delete_embedding(
        self,
        memory_id: str,
    ) -> bool:
        """
        Delete the embedding associated with a memory.
        """

        conn = self._connect()

        cursor = conn.execute(
            """
            DELETE FROM memory_embeddings
            WHERE memory_id = ?
            """,
            (memory_id,),
        )

        conn.commit()

        if self._connection is None:
            conn.close()

        return cursor.rowcount > 0

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
        """

        query = query.strip()

        if not query:
            return []

        limit = max(
            1,
            int(limit),
        )

        normalized_query = query.lower()

        # -----------------------------------------------------
        # Tokenization
        # -----------------------------------------------------

        def tokenize(
            text: str,
        ) -> list[str]:
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

        def score(
            memory: Memory,
        ) -> float:
            content = (
                memory.content.lower()
            )

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

            # Exact phrase.
            if normalized_query in content:
                score_value += 12.0

            # Exact content tokens.
            score_value += (
                len(
                    exact_content_hits
                )
                * 4.0
            )

            # Exact tag tokens.
            score_value += (
                len(
                    exact_tag_hits
                )
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

            # Memory quality.
            score_value += (
                memory.importance
                * 1.5
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

        return ranked[
            :limit
        ]