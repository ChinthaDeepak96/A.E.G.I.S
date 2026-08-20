# A.E.G.I.S. v0.4.0 - Persistent Memory Foundation

This milestone adds the persistent memory foundation without yet coupling it
to the live conversation loop.

Included:
- SQLite persistence
- working, episodic, semantic, and preference memory types
- importance, confidence, and sensitivity metadata
- explicit and policy-based storage
- lexical retrieval
- memory deletion and clearing
- tests

Not included yet:
- embeddings/vector database
- automatic memory extraction from conversations
- memory consolidation
- cloud synchronization
- full Guardian integration

Copy `memory/` into the A.E.G.I.S. project root.

Add `data/` to `.gitignore` because the database contains personal memory.

The intended integration is through `MemoryManager`; A.E.G.I.S. should not
access SQLite directly.
