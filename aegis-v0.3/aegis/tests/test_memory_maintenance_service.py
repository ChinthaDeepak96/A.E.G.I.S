from datetime import datetime, timedelta, timezone

from memory.health import MemoryHealth
from memory.maintenance import (
    MaintenanceAction,
)
from memory.maintenance_executor import (
    MaintenanceExecutionResult,
)
from memory.maintenance_service import (
    MaintenanceBatchEvaluation,
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
        / "maintenance_service.db"
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


# =========================================================
# CONSTRUCTION
# =========================================================


def test_service_uses_manager_policy(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    assert (
        service.policy
        is manager.policy
    )

    assert (
        service.executor.policy
        is manager.policy
    )


def test_invalid_manager_is_rejected():
    try:
        MemoryMaintenanceService(
            None
        )

        assert False

    except TypeError:
        pass


# =========================================================
# EVALUATION
# =========================================================


def test_evaluate_returns_complete_result(
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

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate(
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

    assert result.health is not None

    assert result.proposal is not None

    assert (
        result.proposal.memory_id
        == memory.id
    )

    assert (
        result.decision
        == MemoryPolicyDecision.ALLOW
    )


def test_evaluate_healthy_memory_requires_no_action(
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

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate(
        memory.id
    )

    assert result is not None

    assert (
        result.health.health
        == MemoryHealth.HEALTHY
    )

    assert (
        result.proposal.action
        == MaintenanceAction.NO_ACTION
    )

    assert (
        result.proposal
        .requires_confirmation
        is False
    )


def test_evaluate_aging_memory_requires_review(
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
        days=45
    )

    memory.created_at = old
    memory.updated_at = old

    manager.store.save(
        memory
    )

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate(
        memory.id
    )

    assert result is not None

    assert (
        result.health.health
        == MemoryHealth.AGING
    )

    assert (
        result.proposal.action
        == MaintenanceAction.REVIEW
    )

    assert (
        result.decision
        == MemoryPolicyDecision.ALLOW
    )


def test_evaluate_stale_memory_requires_confirmation(
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

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate(
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
        result.proposal
        .requires_confirmation
        is True
    )

    assert (
        result.decision
        == MemoryPolicyDecision.CONFIRM
    )


def test_evaluate_missing_memory_returns_none(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate(
        "missing-memory"
    )

    assert result is None


def test_evaluate_empty_memory_id_returns_none(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate(
        "   "
    )

    assert result is None


# =========================================================
# READ-ONLY EVALUATION
# =========================================================


def test_evaluate_does_not_modify_memory(
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

    service = MemoryMaintenanceService(
        manager
    )

    service.evaluate(
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


# =========================================================
# BATCH EVALUATION
# =========================================================


def test_evaluate_many_preserves_order(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    first = manager.remember(
        "First A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
    )

    second = manager.remember(
        "Second A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
    )

    assert first is not None
    assert second is not None

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate_many(
        [
            second.id,
            first.id,
        ]
    )

    assert isinstance(
        result,
        MaintenanceBatchEvaluation,
    )

    assert len(
        result.evaluations
    ) == 2

    assert (
        result.evaluations[0]
        .memory.id
        == second.id
    )

    assert (
        result.evaluations[1]
        .memory.id
        == first.id
    )


def test_evaluate_many_ignores_missing_ids(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
    )

    assert memory is not None

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate_many(
        [
            "missing-memory",
            memory.id,
        ]
    )

    assert len(
        result.evaluations
    ) == 1

    assert (
        result.evaluations[0]
        .memory.id
        == memory.id
    )


def test_evaluate_many_empty_list(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    result = service.evaluate_many(
        []
    )

    assert isinstance(
        result,
        MaintenanceBatchEvaluation,
    )

    assert (
        result.evaluations
        == []
    )


# =========================================================
# EXECUTION
# =========================================================


def test_execute_no_action_does_not_mutate(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
    )

    assert memory is not None

    service = MemoryMaintenanceService(
        manager
    )

    result = service.execute(
        memory.id
    )

    assert isinstance(
        result,
        MaintenanceExecutionResult,
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


def test_execute_stale_requires_confirmation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
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

    service = MemoryMaintenanceService(
        manager
    )

    result = service.execute(
        memory.id
    )

    assert result is not None

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


def test_execute_stale_after_confirmation(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
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

    service = MemoryMaintenanceService(
        manager
    )

    result = service.execute(
        memory.id,
        confirmed=True,
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


def test_execute_missing_memory_returns_none(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    result = service.execute(
        "missing-memory"
    )

    assert result is None


# =========================================================
# PROPOSAL EXECUTION
# =========================================================


def test_execute_proposal_delegates_to_executor(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
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

    service = MemoryMaintenanceService(
        manager
    )

    evaluation = service.evaluate(
        memory.id
    )

    assert evaluation is not None

    result = service.execute_proposal(
        evaluation.proposal,
        confirmed=True,
    )

    assert (
        result.executed
        is True
    )

    assert result.memory is not None

    assert (
        result.memory.status
        == MemoryStatus.STALE
    )


# =========================================================
# CONVENIENCE API
# =========================================================


def test_review_alias_matches_evaluate(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
    )

    assert memory is not None

    service = MemoryMaintenanceService(
        manager
    )

    evaluated = service.evaluate(
        memory.id
    )

    reviewed = service.review(
        memory.id
    )

    assert evaluated is not None
    assert reviewed is not None

    assert (
        evaluated.proposal.action
        == reviewed.proposal.action
    )

    assert (
        evaluated.decision
        == reviewed.decision
    )


def test_pending_action_returns_action(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
    )

    assert memory is not None

    service = MemoryMaintenanceService(
        manager
    )

    action = service.pending_action(
        memory.id
    )

    assert (
        action
        == MaintenanceAction.NO_ACTION
    )


def test_pending_action_missing_memory_returns_none(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    assert (
        service.pending_action(
            "missing-memory"
        )
        is None
    )


def test_requires_confirmation_for_stale_memory(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
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

    service = MemoryMaintenanceService(
        manager
    )

    assert (
        service.requires_confirmation(
            memory.id
        )
        is True
    )


def test_requires_confirmation_for_healthy_memory_is_false(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    memory = manager.remember(
        "A.E.G.I.S. memory.",
        importance=0.9,
        confidence=0.9,
    )

    assert memory is not None

    service = MemoryMaintenanceService(
        manager
    )

    assert (
        service.requires_confirmation(
            memory.id
        )
        is False
    )


def test_requires_confirmation_missing_memory_is_false(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    assert (
        service.requires_confirmation(
            "missing-memory"
        )
        is False
    )


# =========================================================
# INPUT VALIDATION
# =========================================================


def test_evaluate_rejects_non_string_memory_id(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    try:
        service.evaluate(
            None
        )

        assert False

    except TypeError:
        pass


def test_evaluate_many_rejects_non_list(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    try:
        service.evaluate_many(
            None
        )

        assert False

    except TypeError:
        pass


def test_evaluate_many_rejects_invalid_usage_map(
    tmp_path,
):
    manager = make_manager(
        tmp_path
    )

    service = MemoryMaintenanceService(
        manager
    )

    try:
        service.evaluate_many(
            [],
            usages={
                "memory": "invalid"
            },
        )

        assert False

    except TypeError:
        pass