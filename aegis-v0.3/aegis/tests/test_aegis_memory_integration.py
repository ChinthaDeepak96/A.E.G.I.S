from datetime import datetime, timedelta, timezone

from core.aegis import AEGIS
from core.llm_client import (
    LLMResponse,
    MockClient,
    TextBlock,
)
from memory.manager import MemoryManager
from memory.maintenance_service import (
    MemoryMaintenanceService,
)
from memory.models import (
    MemoryStatus,
)


def make_manager(tmp_path):
    database = (
        tmp_path
        / "aegis_memory_integration.db"
    )

    return MemoryManager(
        database_path=str(database)
    )


def make_response(text):
    return LLMResponse(
        content=[
            TextBlock(
                text=text
            )
        ],
        stop_reason="end_turn",
    )


def make_aegis(
    tmp_path,
    *,
    client=None,
):
    manager = make_manager(
        tmp_path
    )

    if client is None:
        client = MockClient(
            responses=[
                make_response(
                    "Test response."
                )
            ]
        )

    aegis = AEGIS(
        client,
        memory_manager=manager,
    )

    return aegis, manager, client


def make_old_timestamp(*, days):
    return (
        datetime.now(
            timezone.utc
        )
        - timedelta(days=days)
    ).isoformat()


# =========================================================
# MEMORY MANAGER INTEGRATION
# =========================================================


def test_aegis_exposes_injected_memory_manager(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    assert (
        aegis.memory
        is manager
    )


def test_aegis_memory_manager_exposes_maintenance(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    assert isinstance(
        aegis.memory.maintenance,
        MemoryMaintenanceService,
    )

    assert (
        aegis.memory.maintenance.manager
        is manager
    )


def test_aegis_without_memory_has_no_memory_subsystem():
    client = MockClient(
        responses=[
            make_response(
                "Test response."
            )
        ]
    )

    aegis = AEGIS(
        client
    )

    assert aegis.memory is None


# =========================================================
# EXPLICIT MEMORY COMMANDS
# =========================================================


def test_aegis_explicit_memory_command_persists_memory(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    result = aegis.respond(
        "/memory remember "
        "I prefer concise technical explanations."
    )

    assert (
        "Memory saved."
        in result
    )

    memories = manager.recent()

    assert len(
        memories
    ) == 1

    assert (
        memories[0].content
        == "I prefer concise technical explanations."
    )


def test_aegis_memory_recall_command_uses_manager(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    result = aegis.respond(
        "/memory recall Gemma 4"
    )

    assert (
        "A.E.G.I.S. uses Gemma 4."
        in result
    )


def test_aegis_memory_recent_command_uses_manager(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. persistent memory.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    result = aegis.respond(
        "/memory recent"
    )

    assert (
        "A.E.G.I.S. persistent memory."
        in result
    )


# =========================================================
# MEMORY CONTEXT
# =========================================================


def test_aegis_builds_memory_context_from_manager(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    memory = manager.remember(
        "The user's project is called A.E.G.I.S.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    context = (
        aegis._build_memory_context(
            "What is my project called?"
        )
    )

    assert (
        "A.E.G.I.S."
        in context
    )


def test_aegis_without_memory_returns_empty_context():
    client = MockClient()

    aegis = AEGIS(
        client
    )

    context = (
        aegis._build_memory_context(
            "What is my project?"
        )
    )

    assert context == ""


# =========================================================
# NORMAL CONVERSATION + MEMORY
# =========================================================


def test_normal_conversation_can_use_persistent_memory(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    client = MockClient(
        responses=[
            make_response(
                "I have the relevant context."
            )
        ]
    )

    aegis = AEGIS(
        client,
        memory_manager=manager,
    )

    result = aegis.respond(
        "What model does A.E.G.I.S. use?"
    )

    assert (
        result
        == "I have the relevant context."
    )

    assert len(
        client.calls
    ) >= 1

    first_call = client.calls[0]

    system_prompt = first_call[
        "system"
    ]

    assert (
        "Relevant persistent memory"
        in system_prompt
    )

    assert (
        "Gemma 4"
        in system_prompt
    )


# =========================================================
# AUTOMATIC EXTRACTION
# =========================================================


def test_aegis_automatic_extraction_uses_memory_manager(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    class FakeExtractor:
        def __init__(self):
            self.called = False
            self.messages = None

        def extract(
            self,
            messages,
        ):
            self.called = True
            self.messages = messages

            from memory.extractor import (
                MemoryCandidate,
            )
            from memory.models import (
                MemoryType,
            )

            return [
                MemoryCandidate(
                    content=(
                        "The user is building "
                        "A.E.G.I.S."
                    ),
                    memory_type=(
                        MemoryType.SEMANTIC
                    ),
                    importance=0.9,
                    confidence=0.9,
                    sensitivity=0.0,
                    tags=[
                        "project"
                    ],
                )
            ]

    extractor = FakeExtractor()

    client = MockClient(
        responses=[
            make_response(
                "Response one."
            ),
            make_response(
                "Response two."
            ),
        ]
    )

    aegis = AEGIS(
        client,
        memory_manager=manager,
        memory_extractor=extractor,
    )

    aegis.respond(
        "I am building A.E.G.I.S."
    )

    aegis.respond(
        "It is my AI assistant project."
    )

    # The extractor must actually run.
    assert (
        extractor.called
        is True
    )

    assert (
        extractor.messages
        is not None
    )

    memories = manager.recent()

    assert any(
        memory.content
        == "The user is building A.E.G.I.S."
        for memory in memories
    )


def test_automatic_extraction_does_not_bypass_policy(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    class LowQualityExtractor:
        def extract(
            self,
            messages,
        ):
            from memory.extractor import (
                MemoryCandidate,
            )
            from memory.models import (
                MemoryType,
            )

            return [
                MemoryCandidate(
                    content=(
                        "Low quality candidate."
                    ),
                    memory_type=(
                        MemoryType.EPISODIC
                    ),
                    importance=0.2,
                    confidence=0.2,
                    sensitivity=0.0,
                    tags=[],
                )
            ]

    client = MockClient(
        responses=[
            make_response(
                "Response one."
            ),
            make_response(
                "Response two."
            ),
        ]
    )

    aegis = AEGIS(
        client,
        memory_manager=manager,
        memory_extractor=LowQualityExtractor(),
    )

    aegis.respond(
        "This is a conversation."
    )

    aegis.respond(
        "Another conversation turn."
    )

    memories = manager.recent()

    assert not any(
        memory.content
        == "Low quality candidate."
        for memory in memories
    )


def test_extraction_failure_does_not_break_conversation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    class BrokenExtractor:
        def extract(
            self,
            messages,
        ):
            raise RuntimeError(
                "simulated extraction failure"
            )

    client = MockClient(
        responses=[
            make_response(
                "First response."
            ),
            make_response(
                "Second response."
            ),
        ]
    )

    aegis = AEGIS(
        client,
        memory_manager=manager,
        memory_extractor=BrokenExtractor(),
    )

    first = aegis.respond(
        "First turn."
    )

    second = aegis.respond(
        "Second turn."
    )

    assert (
        first
        == "First response."
    )

    assert (
        second
        == "Second response."
    )


# =========================================================
# RESET BEHAVIOR
# =========================================================


def test_reset_clears_conversation_and_extraction_counter(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    aegis.respond(
        "First conversation turn."
    )

    assert (
        aegis._memory_extraction_turns
        >= 1
    )

    memory = manager.remember(
        "Persistent memory survives reset.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    aegis.reset()

    assert (
        aegis._history
        == []
    )

    assert (
        aegis._memory_extraction_turns
        == 0
    )

    persisted = manager.get(
        memory.id
    )

    assert persisted is not None

    assert (
        persisted.content
        == "Persistent memory survives reset."
    )


# =========================================================
# MAINTENANCE INTEGRATION
# =========================================================


def test_aegis_memory_can_evaluate_maintenance(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    memory = manager.remember(
        "Old A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    old = make_old_timestamp(
        days=120
    )

    memory.created_at = old
    memory.updated_at = old

    manager.store.save(
        memory
    )

    evaluation = (
        aegis.memory.maintenance.evaluate(
            memory.id
        )
    )

    assert evaluation is not None

    assert (
        evaluation.proposal
        .requires_confirmation
        is True
    )


def test_aegis_maintenance_requires_confirmation(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    memory = manager.remember(
        "Old A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    old = make_old_timestamp(
        days=120
    )

    memory.created_at = old
    memory.updated_at = old

    manager.store.save(
        memory
    )

    result = (
        aegis.memory.maintenance.execute(
            memory.id
        )
    )

    assert result is not None

    assert (
        result.executed
        is False
    )

    persisted = manager.get(
        memory.id
    )

    assert persisted is not None

    assert (
        persisted.status
        == MemoryStatus.ACTIVE
    )


def test_aegis_maintenance_can_execute_after_confirmation(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    memory = manager.remember(
        "Old A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    old = make_old_timestamp(
        days=120
    )

    memory.created_at = old
    memory.updated_at = old

    manager.store.save(
        memory
    )

    result = (
        aegis.memory.maintenance.execute(
            memory.id,
            confirmed=True,
        )
    )

    assert result is not None

    assert (
        result.executed
        is True
    )

    persisted = manager.get(
        memory.id
    )

    assert persisted is not None

    assert (
        persisted.status
        == MemoryStatus.STALE
    )


# =========================================================
# MEMORY LIFECYCLE SAFETY
# =========================================================


def test_normal_conversation_does_not_automatically_archive_memory(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "Important A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    old = make_old_timestamp(
        days=365
    )

    memory.created_at = old
    memory.updated_at = old

    manager.store.save(
        memory
    )

    client = MockClient(
        responses=[
            make_response(
                "Normal response."
            )
        ]
    )

    aegis = AEGIS(
        client,
        memory_manager=manager,
    )

    aegis.respond(
        "Continue our conversation."
    )

    persisted = manager.get(
        memory.id
    )

    assert persisted is not None

    assert (
        persisted.status
        == MemoryStatus.ACTIVE
    )


def test_memory_context_is_not_written_back_to_history_as_memory(
    tmp_path,
):
    aegis, manager, _ = make_aegis(
        tmp_path
    )

    memory = manager.remember(
        "Persistent context.",
        importance=0.9,
        confidence=0.9,
        explicit=True,
    )

    assert memory is not None

    aegis.respond(
        "Tell me something."
    )

    history = aegis._history

    assert all(
        item.get("role")
        != "memory"
        for item in history
    )