from datetime import datetime, timedelta, timezone

from memory.health import MemoryHealth
from memory.maintenance import (
    MaintenanceAction,
    MemoryMaintenancePlanner,
)
from memory.models import Memory


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