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

def test_never_retrieved_memory_has_zero_usage_score():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Unused memory.",
        importance=0.8,
        confidence=0.9,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    from memory.usage import MemoryUsage

    usage = MemoryUsage()

    report = analyzer.analyze(
        memory,
        usage=usage,
        now=now,
    )

    assert (
        report.retrieval_count
        == 0
    )

    assert (
        report.access_count
        == 0
    )

    assert (
        report.has_been_retrieved
        is False
    )

    assert (
        report.days_since_retrieval
        is None
    )


def test_retrieved_memory_reports_usage():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Frequently used memory.",
        importance=0.8,
        confidence=0.9,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    from memory.usage import MemoryUsage

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=now
    )

    usage.record_retrieval(
        timestamp=now
    )

    report = analyzer.analyze(
        memory,
        usage=usage,
        now=now,
    )

    assert (
        report.retrieval_count
        == 2
    )

    assert (
        report.access_count
        == 2
    )

    assert (
        report.has_been_retrieved
        is True
    )

    assert (
        report.days_since_retrieval
        == 0.0
    )


def test_recent_retrieval_improves_health_score():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Used memory.",
        importance=0.8,
        confidence=0.9,
        created_at=(
            now
            - timedelta(days=45)
        ).isoformat(),
        updated_at=(
            now
            - timedelta(days=45)
        ).isoformat(),
    )

    from memory.usage import MemoryUsage

    unused = MemoryUsage()

    used = MemoryUsage()

    used.record_retrieval(
        timestamp=now
    )

    unused_report = analyzer.analyze(
        memory,
        usage=unused,
        now=now,
    )

    used_report = analyzer.analyze(
        memory,
        usage=used,
        now=now,
    )

    assert (
        used_report.health_score
        > unused_report.health_score
    )


def test_frequent_retrieval_improves_health_score():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Frequently retrieved memory.",
        importance=0.8,
        confidence=0.9,
        created_at=(
            now
            - timedelta(days=45)
        ).isoformat(),
        updated_at=(
            now
            - timedelta(days=45)
        ).isoformat(),
    )

    from memory.usage import MemoryUsage

    low_usage = MemoryUsage()

    low_usage.record_retrieval(
        timestamp=now
    )

    high_usage = MemoryUsage()

    for _ in range(10):
        high_usage.record_retrieval(
            timestamp=now
        )

    low_report = analyzer.analyze(
        memory,
        usage=low_usage,
        now=now,
    )

    high_report = analyzer.analyze(
        memory,
        usage=high_usage,
        now=now,
    )

    assert (
        high_report.health_score
        > low_report.health_score
    )


def test_old_memory_remains_stale_candidate_despite_recent_usage():
    analyzer = MemoryHealthAnalyzer()

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

    from memory.usage import MemoryUsage

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=now
    )

    report = analyzer.analyze(
        memory,
        usage=usage,
        now=now,
    )

    assert (
        report.health
        == MemoryHealth.STALE_CANDIDATE
    )

    assert (
        report.health_score
        > 0.0
    )


def test_usage_does_not_modify_memory():
    analyzer = MemoryHealthAnalyzer()

    now = datetime.now(
        timezone.utc
    )

    memory = Memory(
        "Usage analysis test.",
        importance=0.8,
        confidence=0.9,
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    from memory.usage import MemoryUsage

    usage = MemoryUsage()

    usage.record_retrieval(
        timestamp=now
    )

    original_status = memory.status
    original_content = memory.content
    original_updated_at = memory.updated_at

    analyzer.analyze(
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


def test_analyze_many_accepts_usage_map():
    analyzer = MemoryHealthAnalyzer()

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

    from memory.usage import MemoryUsage

    first_usage = MemoryUsage()

    first_usage.record_retrieval(
        timestamp=now
    )

    usages = {
        first.id: first_usage,
    }

    reports = analyzer.analyze_many(
        [
            first,
            second,
        ],
        usages=usages,
        now=now,
    )

    assert len(reports) == 2

    assert (
        reports[0].retrieval_count
        == 1
    )

    assert (
        reports[1].retrieval_count
        == 0
    )


def test_invalid_usage_half_life_is_rejected():
    try:
        MemoryHealthAnalyzer(
            usage_half_life_days=0,
        )
        assert False
    except ValueError:
        pass


def test_invalid_retrieval_saturation_is_rejected():
    try:
        MemoryHealthAnalyzer(
            retrieval_saturation=0,
        )
        assert False
    except ValueError:
        pass