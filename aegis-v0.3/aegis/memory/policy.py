"""
A.E.G.I.S. memory governance policy.

The policy determines whether a memory can become persistent.

Automatic memory extraction is intentionally more restrictive than
explicit/user-directed memory storage.
"""

from __future__ import annotations

from dataclasses import dataclass

from .models import Memory


@dataclass
class MemoryPolicy:
    """
    Controls persistent memory creation.
    """

    automatic_importance_threshold: float = 0.65

    # Preserve the existing public MemoryManager behavior.
    # Automatic candidates are separately restricted by
    # MemoryManager.store_candidate().
    allow_sensitive_memory: bool = True

    minimum_confidence: float = 0.50

    def should_store(
        self,
        memory: Memory,
        explicit: bool = False,
    ) -> bool:
        """
        Decide whether a memory should be persisted.

        Explicit memories bypass the automatic importance/confidence
        requirements.

        Non-explicit memories must satisfy the automatic importance
        and confidence thresholds.
        """

        # Explicit user-directed memory.
        if explicit:
            return True

        # Automatic/general memory storage.
        if memory.confidence < self.minimum_confidence:
            return False

        if memory.sensitivity > 0 and not self.allow_sensitive_memory:
            return False

        if memory.importance < self.automatic_importance_threshold:
            return False

        return True