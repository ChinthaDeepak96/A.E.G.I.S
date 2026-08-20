from datetime import datetime, timedelta, timezone

from memory.health import (
    MemoryHealth,
    MemoryHealthAnalyzer,
)
from memory.models import (
    Memory,
    MemoryStatus,
)


def test_recent_high_confidence_memory_is_healthy():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "A.E.G.I.S. uses Gemma 4.",
        importance=0.9,
        confidence=0.95,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    report = analyzer.analyze(
        memory,
        now=now,
    )

    assert (
        report.health
        == MemoryHealth.HEALTHY
    )

    assert report.age_days == 0.0

    assert report.health_score > 0.8


def test_aging_memory_is_classified_correctly():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=45)
    ).isoformat()

    memory = Memory(
        "An older AEGIS configuration.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    report = analyzer.analyze(
        memory,
        now=now,
    )

    assert (
        report.health
        == MemoryHealth.AGING
    )

    assert report.age_days >= 45.0


def test_old_memory_becomes_stale_candidate():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=120)
    ).isoformat()

    memory = Memory(
        "Old AEGIS configuration.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    report = analyzer.analyze(
        memory,
        now=now,
    )

    assert (
        report.health
        == MemoryHealth.STALE_CANDIDATE
    )


def test_very_old_memory_becomes_archival_candidate():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=200)
    ).isoformat()

    memory = Memory(
        "Very old AEGIS configuration.",
        importance=0.8,
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    report = analyzer.analyze(
        memory,
        now=now,
    )

    assert (
        report.health
        == MemoryHealth.ARCHIVAL_CANDIDATE
    )


def test_low_confidence_memory_can_be_stale_candidate():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Possibly unreliable configuration.",
        importance=0.8,
        confidence=0.3,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    report = analyzer.analyze(
        memory,
        now=now,
    )

    assert (
        report.health
        == MemoryHealth.STALE_CANDIDATE
    )


def test_very_low_confidence_memory_can_be_archival_candidate():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Highly uncertain information.",
        importance=0.8,
        confidence=0.1,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    report = analyzer.analyze(
        memory,
        now=now,
    )

    assert (
        report.health
        == MemoryHealth.ARCHIVAL_CANDIDATE
    )


def test_explicit_stale_memory_is_archival_candidate():
    analyzer = MemoryHealthAnalyzer()

    memory = Memory(
        "Stale AEGIS memory.",
        status=MemoryStatus.STALE,
    )

    report = analyzer.analyze(
        memory
    )

    assert (
        report.health
        == MemoryHealth.ARCHIVAL_CANDIDATE
    )

    assert report.stale is True


def test_archived_memory_is_archival_candidate():
    analyzer = MemoryHealthAnalyzer()

    memory = Memory(
        "Archived AEGIS memory.",
        status=MemoryStatus.ARCHIVED,
    )

    report = analyzer.analyze(
        memory
    )

    assert (
        report.health
        == MemoryHealth.ARCHIVAL_CANDIDATE
    )

    assert report.archived is True


def test_superseded_memory_is_archival_candidate():
    analyzer = MemoryHealthAnalyzer()

    memory = Memory(
        "Old model configuration.",
        status=MemoryStatus.SUPERSEDED,
        superseded_by="replacement-id",
    )

    report = analyzer.analyze(
        memory
    )

    assert (
        report.health
        == MemoryHealth.ARCHIVAL_CANDIDATE
    )

    assert report.superseded is True


def test_health_analysis_does_not_modify_memory():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "A.E.G.I.S. uses Gemma.",
        importance=0.8,
        confidence=0.9,
        created_at=(
            now
            - timedelta(days=120)
        ).isoformat(),
        updated_at=(
            now
            - timedelta(days=120)
        ).isoformat(),
    )

    original_status = memory.status
    original_updated_at = memory.updated_at
    original_content = memory.content

    analyzer.analyze(
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


def test_analyze_many_preserves_input_order():
    analyzer = MemoryHealthAnalyzer()

    memories = [
        Memory("First memory."),
        Memory("Second memory."),
        Memory("Third memory."),
    ]

    reports = analyzer.analyze_many(
        memories
    )

    assert len(reports) == 3

    assert [
        report.memory_id
        for report in reports
    ] == [
        memory.id
        for memory in memories
    ]


def test_custom_thresholds_are_supported():
    analyzer = MemoryHealthAnalyzer(
        aging_days=10,
        stale_days=20,
        archival_days=30,
    )

    now = datetime.now(
        timezone.utc
    )

    old = (
        now
        - timedelta(days=25)
    ).isoformat()

    memory = Memory(
        "Custom threshold memory.",
        confidence=0.9,
        created_at=old,
        updated_at=old,
    )

    report = analyzer.analyze(
        memory,
        now=now,
    )

    assert (
        report.health
        == MemoryHealth.STALE_CANDIDATE
    )


def test_invalid_threshold_order_is_rejected():
    try:
        MemoryHealthAnalyzer(
            aging_days=30,
            stale_days=20,
            archival_days=40,
        )
        assert False
    except ValueError:
        pass