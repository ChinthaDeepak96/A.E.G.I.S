from dataclasses import dataclass

from .models import Memory


@dataclass
class MemoryPolicy:
    automatic_importance_threshold: float = 0.65
    allow_sensitive_memory: bool = True
    minimum_confidence: float = 0.50

    def should_store(self, memory: Memory, explicit: bool = False) -> bool:
        if memory.confidence < self.minimum_confidence:
            return False
        if memory.sensitivity > 0 and not self.allow_sensitive_memory:
            return explicit
        return explicit or memory.importance >= self.automatic_importance_threshold
