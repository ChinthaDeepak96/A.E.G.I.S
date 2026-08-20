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

CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
"""


class SQLiteMemoryStore:
    def __init__(self, path: str | Path = "data/aegis_memory.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def save(self, memory: Memory) -> Memory:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memories
                (id, memory_type, content, importance, confidence,
                 sensitivity, source, tags_json, metadata_json,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                memory.to_record(),
            )
        return memory

    def get(self, memory_id: str) -> Memory | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, memory_type, content, importance, confidence,
                       sensitivity, source, tags_json, metadata_json,
                       created_at, updated_at
                FROM memories WHERE id = ?
                """,
                (memory_id,),
            ).fetchone()
        return Memory.from_row(tuple(row)) if row else None

    def delete(self, memory_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        return cursor.rowcount > 0

    def clear(self) -> int:
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories")
        return cursor.rowcount

    def list(self, memory_type: MemoryType | None = None, limit: int = 100) -> list[Memory]:
        query = """
        SELECT id, memory_type, content, importance, confidence,
               sensitivity, source, tags_json, metadata_json,
               created_at, updated_at
        FROM memories
        """
        params: list[object] = []
        if memory_type:
            query += " WHERE memory_type = ?"
            params.append(memory_type.value)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(max(1, int(limit)))

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [Memory.from_row(tuple(row)) for row in rows]

    def search_text(self, query: str, limit: int = 10) -> list[Memory]:
        terms = [t.strip().lower() for t in query.split() if t.strip()]
        if not terms:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, memory_type, content, importance, confidence,
                       sensitivity, source, tags_json, metadata_json,
                       created_at, updated_at
                FROM memories
                """
            ).fetchall()

        memories = [Memory.from_row(tuple(row)) for row in rows]

        def score(memory: Memory) -> float:
            text = (memory.content + " " + " ".join(memory.tags)).lower()
            hits = sum(1 for term in terms if term in text)
            return hits * 2.0 + memory.importance + memory.confidence

        ranked = sorted(
            (m for m in memories if score(m) > 0),
            key=score,
            reverse=True,
        )
        return ranked[:max(1, int(limit))]
