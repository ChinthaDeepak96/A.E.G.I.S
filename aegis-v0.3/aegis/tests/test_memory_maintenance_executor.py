from datetime import datetime, timezone

from memory.health import MemoryHealth
from memory.maintenance import (
    MaintenanceAction,
    MaintenanceProposal,
)
from memory.maintenance_executor import (
    MaintenanceExecutionResult,
    MemoryMaintenanceExecutor,
)
from memory.manager import MemoryManager
from memory.models import MemoryStatus
from memory.policy import MemoryPolicyDecision


def make_proposal(
    memory_id: str,
    action: MaintenanceAction,
    *,
    requires_confirmation: bool = False,
):
    return MaintenanceProposal(
        memory_id=memory_id,
        action=action,
        health=MemoryHealth.HEALTHY,
        health_score=0.9,
        reason="Test maintenance proposal.",
        requires_confirmation=requires_confirmation,
    )


def make_manager(
    tmp_path,
):
    database = (
        tmp_path
        / "maintenance_executor.db"
    )

    return MemoryManager(
        database_path=str(database)
    )


def test_executor_uses_manager_policy(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    assert executor.policy is manager.policy


def test_no_action_does_not_mutate_memory(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    proposal = make_proposal(
        memory.id,
        MaintenanceAction.NO_ACTION,
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    result = executor.execute(
        proposal
    )

    assert isinstance(
        result,
        MaintenanceExecutionResult,
    )

    assert (
        result.decision
        == MemoryPolicyDecision.ALLOW
    )

    assert (
        result.executed
        is False
    )

    assert (
        result.memory
        is None
    )

    current = manager.get(
        memory.id
    )

    assert current is not None

    assert (
        current.status
        == MemoryStatus.ACTIVE
    )


def test_review_does_not_mutate_memory(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    proposal = make_proposal(
        memory.id,
        MaintenanceAction.REVIEW,
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    result = executor.execute(
        proposal
    )

    assert (
        result.decision
        == MemoryPolicyDecision.ALLOW
    )

    assert (
        result.executed
        is False
    )

    current = manager.get(
        memory.id
    )

    assert current is not None

    assert (
        current.status
        == MemoryStatus.ACTIVE
    )


def test_mark_stale_requires_confirmation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    proposal = make_proposal(
        memory.id,
        MaintenanceAction.MARK_STALE,
        requires_confirmation=True,
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    result = executor.execute(
        proposal
    )

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )

    assert (
        result.executed
        is False
    )

    current = manager.get(
        memory.id
    )

    assert current is not None

    assert (
        current.status
        == MemoryStatus.ACTIVE
    )


def test_mark_stale_executes_after_confirmation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    proposal = make_proposal(
        memory.id,
        MaintenanceAction.MARK_STALE,
        requires_confirmation=True,
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    result = executor.execute(
        proposal,
        confirmed=True,
    )

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )

    assert (
        result.executed
        is True
    )

    assert (
        result.confirmed
        is True
    )

    assert result.memory is not None

    assert (
        result.memory.status
        == MemoryStatus.STALE
    )

    current = manager.get(
        memory.id
    )

    assert current is not None

    assert (
        current.status
        == MemoryStatus.STALE
    )


def test_archive_requires_confirmation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    proposal = make_proposal(
        memory.id,
        MaintenanceAction.ARCHIVE,
        requires_confirmation=True,
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    result = executor.execute(
        proposal
    )

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )

    assert (
        result.executed
        is False
    )

    current = manager.get(
        memory.id
    )

    assert current is not None

    assert (
        current.status
        == MemoryStatus.ACTIVE
    )


def test_archive_executes_after_confirmation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
    )

    assert memory is not None

    proposal = make_proposal(
        memory.id,
        MaintenanceAction.ARCHIVE,
        requires_confirmation=True,
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    result = executor.execute(
        proposal,
        confirmed=True,
    )

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )

    assert (
        result.executed
        is True
    )

    assert (
        result.confirmed
        is True
    )

    assert result.memory is not None

    assert (
        result.memory.status
        == MemoryStatus.ARCHIVED
    )

    current = manager.get(
        memory.id
    )

    assert current is not None

    assert (
        current.status
        == MemoryStatus.ARCHIVED
    )


def test_failed_stale_execution_is_reported(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    proposal = make_proposal(
        "missing-memory",
        MaintenanceAction.MARK_STALE,
        requires_confirmation=True,
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    result = executor.execute(
        proposal,
        confirmed=True,
    )

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )

    assert (
        result.executed
        is False
    )

    assert (
        result.memory
        is None
    )


def test_failed_archive_execution_is_reported(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    proposal = make_proposal(
        "missing-memory",
        MaintenanceAction.ARCHIVE,
        requires_confirmation=True,
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    result = executor.execute(
        proposal,
        confirmed=True,
    )

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )

    assert (
        result.executed
        is False
    )

    assert (
        result.memory
        is None
    )


def test_invalid_proposal_is_rejected(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    try:
        executor.execute(
            None
        )

        assert False

    except TypeError:
        pass


def test_invalid_manager_is_rejected():
    try:
        MemoryMaintenanceExecutor(
            None
        )

        assert False

    except TypeError:
        pass
    