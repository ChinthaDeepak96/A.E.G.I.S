from datetime import datetime, timedelta, timezone

from memory.models import Memory, MemoryType
from memory.scoring import MemoryScorer


def test_relevant_memory_scores_higher_than_unrelated_memory():
    scorer = MemoryScorer()

    relevant = Memory(
        "A.E.G.I.S. uses Gemma 4 through Ollama.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
        confidence=0.9,
    )

    unrelated = Memory(
        "The project has a Guardian risk system.",
        memory_type=MemoryType.SEMANTIC,
        importance=0.8,
        confidence=0.9,
    )

    relevant_score = scorer.score(
        relevant,
        "Gemma Ollama",
    )

    unrelated_score = scorer.score(
        unrelated,
        "Gemma Ollama",
    )

    assert relevant_score > unrelated_score


def test_high_importance_can_improve_score():
    scorer = MemoryScorer()

    low = Memory(
        "A.E.G.I.S. uses Gemma.",
        importance=0.2,
        confidence=0.9,
    )

    high = Memory(
        "A.E.G.I.S. uses Gemma.",
        importance=1.0,
        confidence=0.9,
    )

    assert (
        scorer.score(
            high,
            "Gemma",
        )
        > scorer.score(
            low,
            "Gemma",
        )
    )


def test_high_confidence_can_improve_score():
    scorer = MemoryScorer()

    low = Memory(
        "A.E.G.I.S. uses Gemma.",
        importance=0.8,
        confidence=0.2,
    )

    high = Memory(
        "A.E.G.I.S. uses Gemma.",
        importance=0.8,
        confidence=1.0,
    )

    assert (
        scorer.score(
            high,
            "Gemma",
        )
        > scorer.score(
            low,
            "Gemma",
        )
    )


def test_score_is_between_zero_and_one():
    scorer = MemoryScorer()

    memory = Memory(
        "A.E.G.I.S. uses Gemma.",
        importance=1.0,
        confidence=1.0,
    )

    score = scorer.score(
        memory,
        "Gemma",
    )

    assert 0.0 <= score <= 1.0


def test_empty_query_has_zero_relevance():
    scorer = MemoryScorer()

    memory = Memory(
        "A.E.G.I.S. uses Gemma."
    )

    assert (
        scorer.relevance(
            memory,
            "",
        )
        == 0.0
    )


def test_tags_contribute_to_relevance():
    scorer = MemoryScorer()

    memory = Memory(
        "A.E.G.I.S. uses a local model.",
        tags=["gemma", "ollama"],
    )

    score = scorer.relevance(
        memory,
        "Gemma Ollama",
    )

    assert score > 0.0


def test_phrase_match_receives_bonus():
    scorer = MemoryScorer()

    memory = Memory(
        "A.E.G.I.S. uses Gemma 4."
    )

    score = scorer.relevance(
        memory,
        "A.E.G.I.S. uses Gemma 4.",
    )

    assert score > 0.5


def test_new_memory_is_more_recent_than_old_memory():
    scorer = MemoryScorer()

    now = datetime.now(
        timezone.utc
    )

    new = Memory(
        "New memory.",
        created_at=now.isoformat(),
        updated_at=now.isoformat(),
    )

    old_time = (
        now
        - timedelta(days=180)
    ).isoformat()

    old = Memory(
        "Old memory.",
        created_at=old_time,
        updated_at=old_time,
    )

    assert (
        scorer.recency(new)
        > scorer.recency(old)
    )


def test_rank_returns_best_memory_first():
    scorer = MemoryScorer()

    memories = [
        Memory(
            "The project has a Guardian risk system.",
            importance=0.9,
        ),
        Memory(
            "A.E.G.I.S. uses Gemma 4 through Ollama.",
            importance=0.9,
        ),
    ]

    ranked = scorer.rank(
        memories,
        "Gemma Ollama",
    )

    assert (
        "Gemma"
        in ranked[0].content
    )


def test_score_breakdown_contains_all_components():
    scorer = MemoryScorer()

    memory = Memory(
        "A.E.G.I.S. uses Gemma.",
        importance=0.8,
        confidence=0.9,
    )

    breakdown = scorer.score_breakdown(
        memory,
        "Gemma",
    )

    assert "relevance" in breakdown
    assert "importance" in breakdown
    assert "confidence" in breakdown
    assert "recency" in breakdown
    assert "final" in breakdown


def test_custom_weights_are_normalized():
    scorer = MemoryScorer(
        weights={
            "relevance": 10.0,
            "importance": 0.0,
            "confidence": 0.0,
            "recency": 0.0,
        }
    )

    assert (
        abs(
            scorer.weights["relevance"]
            - 1.0
        )
        < 0.0001
    )