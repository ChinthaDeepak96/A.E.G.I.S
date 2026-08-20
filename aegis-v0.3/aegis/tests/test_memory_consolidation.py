from memory.consolidation import (
    ConsolidationProposal,
    MemoryConsolidator,
)
from memory.models import (
    Memory,
    MemoryStatus,
)
from memory.resolution import ResolutionAction
from memory.conflicts import MemoryRelationship


def test_unrelated_candidate_has_no_existing_memory():
    consolidator = MemoryConsolidator()

    candidate = Memory(
        "My favorite programming language is Python."
    )

    existing = [
        Memory(
            "AEGIS uses Qwen 7B."
        )
    ]

    proposal = consolidator.consolidate(
        candidate,
        existing,
    )

    assert isinstance(
        proposal,
        ConsolidationProposal,
    )

    assert (
        proposal.existing_memory
        is None
    )

    assert (
        proposal.decision.action
        == ResolutionAction.KEEP_BOTH
    )

    assert (
        proposal.decision.relationship
        == MemoryRelationship.UNRELATED
    )


def test_duplicate_candidate_is_rejected():
    consolidator = MemoryConsolidator()

    existing = Memory(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    candidate = Memory(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    proposal = consolidator.consolidate(
        candidate,
        [existing],
    )

    assert (
        proposal.existing_memory
        is existing
    )

    assert (
        proposal.decision.action
        == ResolutionAction.REJECT_CANDIDATE
    )

    assert (
        proposal.decision.relationship
        == MemoryRelationship.DUPLICATE
    )


def test_conflict_preserves_existing_by_default():
    consolidator = MemoryConsolidator()

    existing = Memory(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    candidate = Memory(
        "AEGIS now uses Gemma 4.",
        importance=0.95,
    )

    proposal = consolidator.consolidate(
        candidate,
        [existing],
    )

    assert (
        proposal.existing_memory
        is existing
    )

    assert (
        proposal.decision.action
        == ResolutionAction.KEEP_EXISTING
    )

    assert (
        proposal.decision.relationship
        == MemoryRelationship.CONFLICT
    )

    assert (
        proposal.decision.requires_confirmation
        is True
    )


def test_conflict_can_propose_supersession():
    consolidator = MemoryConsolidator()

    existing = Memory(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    candidate = Memory(
        "AEGIS now uses Gemma 4.",
        importance=0.95,
    )

    proposal = consolidator.consolidate(
        candidate,
        [existing],
        allow_supersession=True,
    )

    assert (
        proposal.decision.action
        == ResolutionAction.SUPERSEDE_EXISTING
    )

    assert (
        proposal.decision.existing_memory_id
        == existing.id
    )

    # Proposal must not mutate the memory.
    assert (
        existing.status
        == MemoryStatus.ACTIVE
    )


def test_stale_memory_is_ignored():
    consolidator = MemoryConsolidator()

    existing = Memory(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    existing.mark_stale()

    candidate = Memory(
        "AEGIS now uses Gemma 4.",
        importance=0.95,
    )

    proposal = consolidator.consolidate(
        candidate,
        [existing],
    )

    assert (
        proposal.existing_memory
        is None
    )

    assert (
        existing.status
        == MemoryStatus.STALE
    )


def test_archived_memory_is_ignored():
    consolidator = MemoryConsolidator()

    existing = Memory(
        "AEGIS uses Qwen 7B.",
        importance=0.9,
    )

    existing.archive()

    candidate = Memory(
        "AEGIS now uses Gemma 4.",
        importance=0.95,
    )

    proposal = consolidator.consolidate(
        candidate,
        [existing],
    )

    assert (
        proposal.existing_memory
        is None
    )


def test_superseded_memory_is_ignored():
    consolidator = MemoryConsolidator()

    replacement = Memory(
        "AEGIS uses Gemma 4."
    )

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    existing.supersede(
        replacement.id
    )

    candidate = Memory(
        "AEGIS now uses Claude."
    )

    proposal = consolidator.consolidate(
        candidate,
        [
            existing,
            replacement,
        ],
    )

    assert (
        proposal.existing_memory
        is replacement
        or proposal.existing_memory
        is None
    )

    assert (
        existing.status
        == MemoryStatus.SUPERSEDED
    )


def test_multiple_relationships_are_returned():
    consolidator = MemoryConsolidator()

    existing_one = Memory(
        "AEGIS uses Qwen 7B."
    )

    existing_two = Memory(
        "AEGIS uses Gemma 4."
    )

    existing_three = Memory(
        "The project has a Guardian system."
    )

    candidate = Memory(
        "AEGIS now uses Claude."
    )

    relationships = (
        consolidator.find_relationships(
            candidate,
            [
                existing_one,
                existing_two,
                existing_three,
            ],
        )
    )

    assert isinstance(
        relationships,
        list,
    )

    assert all(
        memory.status
        == MemoryStatus.ACTIVE
        for memory, _result
        in relationships
    )


def test_consolidation_does_not_modify_existing_memory():
    consolidator = MemoryConsolidator()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    original_status = (
        existing.status
    )

    original_content = (
        existing.content
    )

    candidate = Memory(
        "AEGIS now uses Gemma 4."
    )

    consolidator.consolidate(
        candidate,
        [existing],
        allow_supersession=True,
    )

    assert (
        existing.status
        == original_status
    )

    assert (
        existing.content
        == original_content
    )


def test_candidate_is_not_modified():
    consolidator = MemoryConsolidator()

    existing = Memory(
        "AEGIS uses Qwen 7B."
    )

    candidate = Memory(
        "AEGIS now uses Gemma 4."
    )

    original_status = (
        candidate.status
    )

    original_content = (
        candidate.content
    )

    consolidator.consolidate(
        candidate,
        [existing],
    )

    assert (
        candidate.status
        == original_status
    )

    assert (
        candidate.content
        == original_content
    )


def test_candidate_count_is_reported():
    consolidator = MemoryConsolidator()

    candidate = Memory(
        "My favorite language is Python."
    )

    existing = [
        Memory("AEGIS uses Qwen."),
        Memory("AEGIS uses Gemma."),
        Memory("The Guardian is enabled."),
    ]

    proposal = consolidator.consolidate(
        candidate,
        existing,
    )

    assert (
        proposal.candidates_checked
        == 3
    )