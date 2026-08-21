from datetime import (
    datetime,
    timedelta,
    timezone,
)

from memory.health import MemoryHealth
from memory.maintenance import (
    MaintenanceAction,
    MemoryMaintenancePlanner,
)
from memory.models import Memory
from memory.usage import MemoryUsage


def test_healthy_memory_requires_no_action():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "A.E.G.I.S. uses Gemma.",
        importance=0.9,
        confidence=0.95,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    proposal = planner.propose(
        memory,
        now=now,
    )

    assert (
        proposal.action
        == MaintenanceAction.NO_ACTION
    )

    assert (
        proposal.health
        == MemoryHealth.HEALTHY
    )

    assert (
        proposal.requires_confirmation
        is False
    )


def test_aging_memory_requires_review():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=45)
    ).isoformat()

    memory = Memory(
        "Aging AEGIS memory.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    proposal = planner.propose(
        memory,
        now=now,
    )

    assert (
        proposal.action
        == MaintenanceAction.REVIEW
    )

    assert (
        proposal.health
        == MemoryHealth.AGING
    )

    assert (
        proposal.requires_confirmation
        is False
    )


def test_stale_candidate_proposes_mark_stale():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=120)
    ).isoformat()

    memory = Memory(
        "Old AEGIS memory.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    proposal = planner.propose(
        memory,
        now=now,
    )

    assert (
        proposal.action
        == MaintenanceAction.MARK_STALE
    )

    assert (
        proposal.health
        == MemoryHealth.STALE_CANDIDATE
    )

    assert (
        proposal.requires_confirmation
        is True
    )


def test_archival_candidate_proposes_archive():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=200)
    ).isoformat()

    memory = Memory(
        "Very old AEGIS memory.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    proposal = planner.propose(
        memory,
        now=now,
    )

    assert (
        proposal.action
        == MaintenanceAction.ARCHIVE
    )

    assert (
        proposal.health
        == MemoryHealth.ARCHIVAL_CANDIDATE
    )

    assert (
        proposal.requires_confirmation
        is True
    )


def test_low_confidence_memory_can_propose_stale():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Uncertain AEGIS memory.",
        importance=0.8,
        confidence=0.3,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    proposal = planner.propose(
        memory,
        now=now,
    )

    assert (
        proposal.action
        == MaintenanceAction.MARK_STALE
    )

    assert (
        proposal.requires_confirmation
        is True
    )


def test_very_low_confidence_memory_can_propose_archive():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Highly uncertain memory.",
        importance=0.8,
        confidence=0.1,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    proposal = planner.propose(
        memory,
        now=now,
    )

    assert (
        proposal.action
        == MaintenanceAction.ARCHIVE
    )

    assert (
        proposal.requires_confirmation
        is True
    )


def test_proposal_does_not_modify_memory():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=120)
    ).isoformat()

    memory = Memory(
        "Old memory.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    original_status = memory.status
    original_updated_at = memory.updated_at
    original_content = memory.content

    planner.propose(
        memory,
        now=now,
    )

    assert (
        memory.status
        == original_status
    )

    assert (
        memory.updated_at
        == original_updated_at
    )

    assert (
        memory.content
        == original_content
    )


def test_propose_many_preserves_order():
    planner = MemoryMaintenancePlanner()

    memories = [
        Memory("First memory."),
        Memory("Second memory."),
        Memory("Third memory."),
    ]

    proposals = planner.propose_many(
        memories
    )

    assert len(proposals) == 3

    assert [
        proposal.memory_id
        for proposal in proposals
    ] == [
        memory.id
        for memory in memories
    ]


def test_proposal_contains_health_score():
    planner = MemoryMaintenancePlanner()

    memory = Memory(
        "A.E.G.I.S. memory.",
        importance=0.8,
        confidence=0.9,
    )

    proposal = planner.propose(
        memory
    )

    assert (
        0.0
        <= proposal.health_score
        <= 1.0
    )


def test_invalid_report_is_rejected():
    planner = MemoryMaintenancePlanner()

    try:
        planner.propose_from_report(
            None
        )
        assert False
    except TypeError:
        pass


# =========================================================
# USAGE-AWARE MAINTENANCE TESTS
# =========================================================


def test_proposal_records_retrieval_count():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Frequently used AEGIS memory.",
        importance=0.8,
        confidence=0.9,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=now
    )

    usage.record_retrieval(
        timestamp=now
    )

    usage.record_retrieval(
        timestamp=now
    )

    proposal = planner.propose(
        memory,
        usage=usage,
        now=now,
    )

    assert (
        proposal.retrieval_count
        == 3
    )


def test_proposal_records_access_count():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Accessed AEGIS memory.",
        importance=0.8,
        confidence=0.9,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=now
    )

    usage.record_access(
        timestamp=now
    )

    proposal = planner.propose(
        memory,
        usage=usage,
        now=now,
    )

    assert (
        proposal.access_count
        == 2
    )


def test_proposal_records_days_since_retrieval():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    retrieval_time = (
        now
        - timedelta(days=10)
    )

    memory = Memory(
        "Previously retrieved memory.",
        importance=0.8,
        confidence=0.9,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=retrieval_time
    )

    proposal = planner.propose(
        memory,
        usage=usage,
        now=now,
    )

    assert (
        proposal.days_since_retrieval
        is not None
    )

    assert (
        abs(
            proposal.days_since_retrieval
            - 10.0
        )
        < 0.0001
    )


def test_proposal_records_retrieval_state():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Retrieved memory.",
        importance=0.8,
        confidence=0.9,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=now
    )

    proposal = planner.propose(
        memory,
        usage=usage,
        now=now,
    )

    assert (
        proposal.has_been_retrieved
        is True
    )


def test_recently_retrieved_stale_memory_still_requires_confirmation():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=120)
    ).isoformat()

    memory = Memory(
        "Old but actively used memory.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=now
    )

    proposal = planner.propose(
        memory,
        usage=usage,
        now=now,
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

    assert (
        proposal.retrieval_count
        == 1
    )


def test_usage_does_not_modify_memory():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=120)
    ).isoformat()

    memory = Memory(
        "Old memory.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=now
    )

    original_status = memory.status
    original_content = memory.content
    original_updated_at = memory.updated_at

    planner.propose(
        memory,
        usage=usage,
        now=now,
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


def test_propose_many_accepts_usage_map():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    first = Memory(
        "First memory.",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    second = Memory(
        "Second memory.",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    first_usage = MemoryUsage()

    first_usage.record_retrieval(
        timestamp=now
    )

    usages = {
        first.id: first_usage,
    }

    proposals = planner.propose_many(
        [
            first,
            second,
        ],
        usages=usages,
        now=now,
    )

    assert len(proposals) == 2

    assert (
        proposals[0].retrieval_count
        == 1
    )

    assert (
        proposals[0].has_been_retrieved
        is True
    )

    assert (
        proposals[1].retrieval_count
        == 0
    )

    assert (
        proposals[1].has_been_retrieved
        is False
    )


def test_missing_usage_defaults_to_zero():
    planner = MemoryMaintenancePlanner()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Memory without usage.",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    proposals = planner.propose_many(
        [memory],
        usages={},
        now=now,
    )

    assert len(proposals) == 1

    proposal = proposals[0]

    assert (
        proposal.retrieval_count
        == 0
    )

    assert (
        proposal.access_count
        == 0
    )

    assert (
        proposal.days_since_retrieval
        is None
    )

    assert (
        proposal.has_been_retrieved
        is False
    )


def test_invalid_usage_is_rejected():
    planner = MemoryMaintenancePlanner()

    memory = Memory(
        "Test memory."
    )

    try:
        planner.propose(
            memory,
            usage="invalid",
        )
        assert False
    except TypeError:
        pass


def test_invalid_usage_map_value_is_rejected():
    planner = MemoryMaintenancePlanner()

    memory = Memory(
        "Test memory."
    )

    try:
        planner.propose_many(
            [memory],
            usages={
                memory.id: "invalid",
            },
        )
        assert False
    except TypeError:
        pass