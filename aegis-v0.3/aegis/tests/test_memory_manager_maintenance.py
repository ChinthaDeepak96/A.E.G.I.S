from datetime import datetime, timedelta, timezone

from memory.health import MemoryHealth
from memory.maintenance import (
    MaintenanceAction,
)
from memory.maintenance_service import (
    MaintenanceEvaluation,
    MemoryMaintenanceService,
)
from memory.manager import MemoryManager
from memory.models import MemoryStatus
from memory.policy import MemoryPolicyDecision


def make_manager(
    tmp_path,
):
    database = (
        tmp_path
        / "manager_maintenance.db"
    )

    return MemoryManager(
        database_path=str(database)
    )


def make_old_timestamp(
    *,
    days: int,
) -> str:
    return (
        datetime.now(
            timezone.utc
        )
        - timedelta(days=days)
    ).isoformat()


def test_manager_exposes_maintenance_service(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    assert isinstance(
        manager.maintenance,
        MemoryMaintenanceService,
    )


def test_manager_maintenance_uses_same_manager(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    assert (
        manager.maintenance.manager
        is manager
    )


def test_manager_maintenance_uses_manager_policy(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    assert (
        manager.maintenance.policy
        is manager.policy
    )


def test_manager_can_evaluate_memory_for_maintenance(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
        confidence=0.9,
    )

    assert memory is not None

    result = manager.maintenance.evaluate(
        memory.id
    )

    assert isinstance(
        result,
        MaintenanceEvaluation,
    )

    assert (
        result.memory.id
        == memory.id
    )

    assert (
        result.health.health
        == MemoryHealth.HEALTHY
    )

    assert (
        result.proposal.action
        == MaintenanceAction.NO_ACTION
    )

    assert (
        result.decision
        == MemoryPolicyDecision.ALLOW
    )


def test_manager_can_evaluate_stale_candidate(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
        confidence=0.9,
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

    result = manager.maintenance.evaluate(
        memory.id
    )

    assert result is not None

    assert (
        result.health.health
        == MemoryHealth.STALE_CANDIDATE
    )

    assert (
        result.proposal.action
        == MaintenanceAction.MARK_STALE
    )

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )


def test_manager_maintenance_execute_respects_confirmation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
        confidence=0.9,
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

    blocked = (
        manager.maintenance.execute(
            memory.id
        )
    )

    assert blocked is not None

    assert (
        blocked.decision
        == MemoryPolicyDecision.CONFIRM
    )

    assert (
        blocked.executed
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


def test_manager_maintenance_execute_with_confirmation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
        confidence=0.9,
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
        manager.maintenance.execute(
            memory.id,
            confirmed=True,
        )
    )

    assert result is not None

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )

    assert (
        result.executed
        is True
    )

    current = manager.get(
        memory.id
    )

    assert current is not None

    assert (
        current.status
        == MemoryStatus.STALE
    )


def test_manager_maintenance_missing_memory(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    result = (
        manager.maintenance.evaluate(
            "missing-memory"
        )
    )

    assert result is None

    execution = (
        manager.maintenance.execute(
            "missing-memory"
        )
    )

    assert execution is None


def test_manager_maintenance_is_read_only_during_evaluation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
        confidence=0.9,
    )

    assert memory is not None

    original_status = memory.status
    original_content = memory.content
    original_updated_at = memory.updated_at

    manager.maintenance.evaluate(
        memory.id
    )

    current = manager.get(
        memory.id
    )

    assert current is not None

    assert (
        current.status
        == original_status
    )

    assert (
        current.content
        == original_content
    )

    assert (
        current.updated_at
        == original_updated_at
    )