from datetime import datetime, timedelta, timezone

from memory.health import MemoryHealth
from memory.maintenance import (
    MaintenanceAction,
    MemoryMaintenancePlanner,
)
from memory.maintenance_executor import (
    MemoryMaintenanceExecutor,
)
from memory.models import MemoryStatus
from memory.policy import MemoryPolicyDecision
from memory.manager import MemoryManager


def make_manager(tmp_path):
    database_path = (
        tmp_path
        / "maintenance_integration.db"
    )

    return MemoryManager(
        database_path=str(database_path)
    )


def make_old_timestamp(
    *,
    days: int,
) -> str:
    return (
        datetime.now(timezone.utc)
        - timedelta(days=days)
    ).isoformat()


# =========================================================
# END-TO-END HEALTH → PLANNER
# =========================================================


def test_aging_memory_produces_review_proposal(
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

    old_time = make_old_timestamp(
        days=45
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
    )

    assert (
        proposal.health
        == MemoryHealth.AGING
    )

    assert (
        proposal.action
        == MaintenanceAction.REVIEW
    )

    assert (
        proposal.requires_confirmation
        is False
    )


def test_stale_memory_produces_mark_stale_proposal(
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

    old_time = make_old_timestamp(
        days=120
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
    )

    assert (
        proposal.health
        == MemoryHealth.STALE_CANDIDATE
    )

    assert (
        proposal.action
        == MaintenanceAction.MARK_STALE
    )

    assert (
        proposal.requires_confirmation
        is True
    )


def test_archival_memory_produces_archive_proposal(
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

    old_time = make_old_timestamp(
        days=365
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
    )

    assert (
        proposal.health
        == MemoryHealth.ARCHIVAL_CANDIDATE
    )

    assert (
        proposal.action
        == MaintenanceAction.ARCHIVE
    )

    assert (
        proposal.requires_confirmation
        is True
    )


# =========================================================
# POLICY GATE
# =========================================================


def test_stale_proposal_is_blocked_without_confirmation(
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

    old_time = make_old_timestamp(
        days=120
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
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


def test_stale_proposal_executes_after_confirmation(
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

    old_time = make_old_timestamp(
        days=120
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
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

    assert result.memory is not None

    assert (
        result.memory.status
        == MemoryStatus.STALE
    )

    persisted = manager.get(
        memory.id
    )

    assert persisted is not None

    assert (
        persisted.status
        == MemoryStatus.STALE
    )


def test_archive_proposal_is_blocked_without_confirmation(
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

    old_time = make_old_timestamp(
        days=365
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
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

    persisted = manager.get(
        memory.id
    )

    assert persisted is not None

    assert (
        persisted.status
        == MemoryStatus.ACTIVE
    )


def test_archive_proposal_executes_after_confirmation(
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

    old_time = make_old_timestamp(
        days=365
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
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

    assert result.memory is not None

    assert (
        result.memory.status
        == MemoryStatus.ARCHIVED
    )

    persisted = manager.get(
        memory.id
    )

    assert persisted is not None

    assert (
        persisted.status
        == MemoryStatus.ARCHIVED
    )


# =========================================================
# IMMUTABILITY
# =========================================================


def test_planner_does_not_modify_memory(
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

    old_time = make_old_timestamp(
        days=120
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    original_status = memory.status
    original_content = memory.content
    original_importance = memory.importance
    original_confidence = memory.confidence

    planner = MemoryMaintenancePlanner()

    planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
    )

    assert (
        memory.status
        == original_status
    )

    assert (
        memory.content
        == original_content
    )

    assert (
        memory.importance
        == original_importance
    )

    assert (
        memory.confidence
        == original_confidence
    )


def test_policy_does_not_modify_proposal(
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

    old_time = make_old_timestamp(
        days=365
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
    )

    original_action = proposal.action
    original_reason = proposal.reason
    original_score = proposal.health_score

    manager.policy.review_maintenance(
        proposal
    )

    assert (
        proposal.action
        == original_action
    )

    assert (
        proposal.reason
        == original_reason
    )

    assert (
        proposal.health_score
        == original_score
    )


# =========================================================
# USAGE EVIDENCE
# =========================================================


def test_usage_evidence_survives_planning(
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

    usage = manager.retriever.get_usage(
        memory.id
    )

    assert usage is not None

    usage.record_retrieval()

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        usage=usage,
    )

    assert (
        proposal.retrieval_count
        == 1
    )

    assert (
        proposal.access_count
        == 1
    )

    assert (
        proposal.has_been_retrieved
        is True
    )

    assert (
        proposal.days_since_retrieval
        is not None
    )

def test_usage_evidence_does_not_change_memory(
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

    usage = manager.retriever.get_usage(
        memory.id
    )

    usage.record_retrieval()

    planner = MemoryMaintenancePlanner()

    planner.propose(
        memory,
        usage=usage,
    )

    assert (
        memory.status
        == original_status
    )

    assert (
        memory.content
        == original_content
    )

    assert (
        memory.updated_at
        == original_updated_at
    )


# =========================================================
# FULL PIPELINE
# =========================================================


def test_full_governed_maintenance_pipeline(
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

    old_time = make_old_timestamp(
        days=365
    )

    memory.created_at = old_time
    memory.updated_at = old_time

    manager.store.save(
        memory
    )

    planner = MemoryMaintenancePlanner()

    proposal = planner.propose(
        memory,
        now=datetime.now(
            timezone.utc
        ),
    )

    assert (
        proposal.action
        == MaintenanceAction.ARCHIVE
    )

    assert (
        proposal.requires_confirmation
        is True
    )

    policy_decision = (
        manager.policy.review_maintenance(
            proposal
        )
    )

    assert (
        policy_decision
        == MemoryPolicyDecision.CONFIRM
    )

    executor = MemoryMaintenanceExecutor(
        manager
    )

    blocked = executor.execute(
        proposal
    )

    assert (
        blocked.executed
        is False
    )

    still_active = manager.get(
        memory.id
    )

    assert still_active is not None

    assert (
        still_active.status
        == MemoryStatus.ACTIVE
    )

    executed = executor.execute(
        proposal,
        confirmed=True,
    )

    assert (
        executed.executed
        is True
    )

    archived = manager.get(
        memory.id
    )

    assert archived is not None

    assert (
        archived.status
        == MemoryStatus.ARCHIVED
    )