from .store import SQLiteMemoryStore


class MemoryRetriever:
    """v0.4 lexical retrieval. Embeddings/vector retrieval comes later."""

    def __init__(self, store: SQLiteMemoryStore):
        self.store = store

    def retrieve(self, query: str, limit: int = 8):
        return self.store.search_text(query, limit=limit)
