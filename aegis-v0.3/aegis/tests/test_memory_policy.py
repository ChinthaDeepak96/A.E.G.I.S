from memory.health import MemoryHealth
from memory.maintenance import (
    MaintenanceAction,
    MaintenanceProposal,
)
from memory.models import Memory
from memory.policy import (
    MemoryPolicy,
    MemoryPolicyDecision,
)


def make_memory(
    *,
    importance=0.8,
    confidence=0.9,
    sensitivity=0.0,
):
    return Memory(
        "A.E.G.I.S. uses Gemma 4.",
        importance=importance,
        confidence=confidence,
        sensitivity=sensitivity,
    )


def make_proposal(
    action: MaintenanceAction,
    *,
    requires_confirmation: bool = False,
):
    return MaintenanceProposal(
        memory_id="test-memory",
        action=action,
        health=MemoryHealth.HEALTHY,
        health_score=0.9,
        reason="Test proposal.",
        requires_confirmation=requires_confirmation,
    )


# =========================================================
# PERSISTENCE POLICY
# =========================================================


def test_default_policy_values():
    policy = MemoryPolicy()

    assert (
        policy.automatic_importance_threshold
        == 0.65
    )

    assert (
        policy.minimum_confidence
        == 0.50
    )

    assert (
        policy.allow_sensitive_memory
        is True
    )


def test_explicit_memory_bypasses_automatic_thresholds():
    policy = MemoryPolicy()

    memory = make_memory(
        importance=0.1,
        confidence=0.1,
    )

    assert (
        policy.should_store(
            memory,
            explicit=True,
        )
        is True
    )


def test_automatic_memory_above_thresholds_is_allowed():
    policy = MemoryPolicy()

    memory = make_memory(
        importance=0.8,
        confidence=0.9,
    )

    assert (
        policy.should_store(
            memory,
            explicit=False,
        )
        is True
    )


def test_low_confidence_automatic_memory_is_rejected():
    policy = MemoryPolicy()

    memory = make_memory(
        importance=0.9,
        confidence=0.49,
    )

    assert (
        policy.should_store(
            memory,
            explicit=False,
        )
        is False
    )


def test_low_importance_automatic_memory_is_rejected():
    policy = MemoryPolicy()

    memory = make_memory(
        importance=0.64,
        confidence=0.9,
    )

    assert (
        policy.should_store(
            memory,
            explicit=False,
        )
        is False
    )


def test_confidence_boundary_is_allowed():
    policy = MemoryPolicy()

    memory = make_memory(
        importance=0.65,
        confidence=0.50,
    )

    assert (
        policy.should_store(
            memory,
            explicit=False,
        )
        is True
    )


def test_importance_boundary_is_allowed():
    policy = MemoryPolicy()

    memory = make_memory(
        importance=0.65,
        confidence=0.50,
    )

    assert (
        policy.should_store(
            memory,
            explicit=False,
        )
        is True
    )


def test_sensitive_memory_allowed_by_default():
    policy = MemoryPolicy(
        allow_sensitive_memory=True
    )

    memory = make_memory(
        importance=0.8,
        confidence=0.9,
        sensitivity=0.8,
    )

    assert (
        policy.should_store(
            memory,
            explicit=False,
        )
        is True
    )


def test_sensitive_memory_can_be_rejected():
    policy = MemoryPolicy(
        allow_sensitive_memory=False
    )

    memory = make_memory(
        importance=0.8,
        confidence=0.9,
        sensitivity=0.8,
    )

    assert (
        policy.should_store(
            memory,
            explicit=False,
        )
        is False
    )


def test_explicit_sensitive_memory_still_bypasses_policy():
    policy = MemoryPolicy(
        allow_sensitive_memory=False
    )

    memory = make_memory(
        importance=0.1,
        confidence=0.1,
        sensitivity=1.0,
    )

    assert (
        policy.should_store(
            memory,
            explicit=True,
        )
        is True
    )


def test_policy_does_not_modify_memory():
    policy = MemoryPolicy()

    memory = make_memory(
        importance=0.8,
        confidence=0.9,
    )

    original_content = memory.content
    original_importance = memory.importance
    original_confidence = memory.confidence
    original_sensitivity = memory.sensitivity
    original_updated_at = memory.updated_at

    policy.should_store(
        memory,
        explicit=False,
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

    assert (
        memory.sensitivity
        == original_sensitivity
    )

    assert (
        memory.updated_at
        == original_updated_at
    )


def test_custom_importance_threshold_is_supported():
    policy = MemoryPolicy(
        automatic_importance_threshold=0.80
    )

    low = make_memory(
        importance=0.79,
        confidence=0.9,
    )

    high = make_memory(
        importance=0.80,
        confidence=0.9,
    )

    assert (
        policy.should_store(
            low,
            explicit=False,
        )
        is False
    )

    assert (
        policy.should_store(
            high,
            explicit=False,
        )
        is True
    )


def test_custom_confidence_threshold_is_supported():
    policy = MemoryPolicy(
        minimum_confidence=0.80
    )

    low = make_memory(
        importance=0.9,
        confidence=0.79,
    )

    high = make_memory(
        importance=0.9,
        confidence=0.80,
    )

    assert (
        policy.should_store(
            low,
            explicit=False,
        )
        is False
    )

    assert (
        policy.should_store(
            high,
            explicit=False,
        )
        is True
    )


# =========================================================
# MAINTENANCE GOVERNANCE
# =========================================================


def test_no_action_maintenance_is_allowed():
    policy = MemoryPolicy()

    proposal = make_proposal(
        MaintenanceAction.NO_ACTION
    )

    assert (
        policy.review_maintenance(
            proposal
        )
        == MemoryPolicyDecision.ALLOW
    )


def test_review_maintenance_is_allowed():
    policy = MemoryPolicy()

    proposal = make_proposal(
        MaintenanceAction.REVIEW
    )

    assert (
        policy.review_maintenance(
            proposal
        )
        == MemoryPolicyDecision.ALLOW
    )


def test_mark_stale_requires_confirmation():
    policy = MemoryPolicy()

    proposal = make_proposal(
        MaintenanceAction.MARK_STALE,
        requires_confirmation=True,
    )

    assert (
        policy.review_maintenance(
            proposal
        )
        == MemoryPolicyDecision.CONFIRM
    )


def test_archive_requires_confirmation():
    policy = MemoryPolicy()

    proposal = make_proposal(
        MaintenanceAction.ARCHIVE,
        requires_confirmation=True,
    )

    assert (
        policy.review_maintenance(
            proposal
        )
        == MemoryPolicyDecision.CONFIRM
    )


def test_maintenance_policy_does_not_modify_proposal():
    policy = MemoryPolicy()

    proposal = make_proposal(
        MaintenanceAction.ARCHIVE,
        requires_confirmation=True,
    )

    original_action = proposal.action
    original_reason = proposal.reason
    original_score = proposal.health_score

    policy.review_maintenance(
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


def test_automatic_maintenance_helper_matches_allow():
    policy = MemoryPolicy()

    proposal = make_proposal(
        MaintenanceAction.REVIEW
    )

    assert (
        policy.allows_automatic_maintenance(
            proposal
        )
        is True
    )


def test_confirmation_helper_matches_confirmation():
    policy = MemoryPolicy()

    proposal = make_proposal(
        MaintenanceAction.ARCHIVE,
        requires_confirmation=True,
    )

    assert (
        policy.requires_maintenance_confirmation(
            proposal
        )
        is True
    )


def test_denial_helper_matches_denial():
    policy = MemoryPolicy()

    proposal = make_proposal(
        MaintenanceAction.NO_ACTION
    )

    assert (
        policy.denies_maintenance(
            proposal
        )
        is False
    )


def test_invalid_maintenance_proposal_is_rejected():
    policy = MemoryPolicy()

    try:
        policy.review_maintenance(
            None
        )

        assert False

    except TypeError:
        pass