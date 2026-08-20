from memory.models import Memory
from memory.scoring import MemoryScorer


def test_semantic_relevance_returns_value_between_zero_and_one():
    scorer = MemoryScorer()

    memory = Memory(
        "AEGIS uses Gemma 4 through Ollama.",
    )

    score = scorer.semantic_relevance(
        memory,
        "What language model does AEGIS use?",
    )

    assert 0.0 <= score <= 1.0


def test_semantically_related_memory_scores_higher():
    scorer = MemoryScorer()

    related = Memory(
        "AEGIS uses Gemma 4 through Ollama.",
    )

    unrelated = Memory(
        "My favorite programming language is Python.",
    )

    related_score = scorer.semantic_relevance(
        related,
        "What language model does AEGIS use?",
    )

    unrelated_score = scorer.semantic_relevance(
        unrelated,
        "What language model does AEGIS use?",
    )

    assert related_score > unrelated_score


def test_semantic_relevance_empty_query_is_zero():
    scorer = MemoryScorer()

    memory = Memory(
        "AEGIS uses Gemma 4.",
    )

    assert (
        scorer.semantic_relevance(
            memory,
            "",
        )
        == 0.0
    )


def test_score_breakdown_contains_semantic_components():
    scorer = MemoryScorer()

    memory = Memory(
        "AEGIS uses Gemma 4.",
        importance=0.8,
        confidence=0.9,
    )

    breakdown = scorer.score_breakdown(
        memory,
        "What model does AEGIS use?",
    )

    assert (
        "semantic_relevance"
        in breakdown
    )

    assert (
        "lexical_relevance"
        in breakdown
    )

    assert (
        "final"
        in breakdown
    )


def test_semantic_signal_can_rank_related_memory():
    scorer = MemoryScorer()

    related = Memory(
        "AEGIS currently runs Gemma 4 locally.",
        importance=0.8,
        confidence=0.9,
    )

    unrelated = Memory(
        "The project has a Guardian risk system.",
        importance=0.8,
        confidence=0.9,
    )

    ranked = scorer.rank(
        [
            unrelated,
            related,
        ],
        "Which AI model does AEGIS run?",
    )

    assert ranked[0].content == (
        "AEGIS currently runs Gemma 4 locally."
    )