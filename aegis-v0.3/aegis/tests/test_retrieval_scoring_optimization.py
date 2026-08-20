from memory.models import Memory
from memory.retrieval import RetrievalCandidate
from memory.scoring import MemoryScorer


def test_precomputed_semantic_score_is_used():
    scorer = MemoryScorer()

    memory = Memory(
        "AEGIS uses Gemma 4.",
        importance=0.8,
        confidence=0.9,
    )

    low = scorer.score(
        memory,
        "What model does AEGIS use?",
        semantic_score=0.0,
    )

    high = scorer.score(
        memory,
        "What model does AEGIS use?",
        semantic_score=0.9,
    )

    assert high > low


def test_zero_cosine_similarity_is_not_half_relevance():
    assert (
        MemoryScorer._semantic_similarity_to_score(
            0.0
        )
        == 0.0
    )


def test_negative_cosine_similarity_has_zero_relevance():
    assert (
        MemoryScorer._semantic_similarity_to_score(
            -0.5
        )
        == 0.0
    )


def test_positive_cosine_similarity_is_preserved():
    assert (
        MemoryScorer._semantic_similarity_to_score(
            0.3883
        )
        == 0.3883
    )


def test_semantic_score_is_clamped():
    assert (
        MemoryScorer._semantic_similarity_to_score(
            2.0
        )
        == 1.0
    )


def test_retrieval_candidate_is_immutable():
    memory = Memory(
        "AEGIS uses Gemma 4."
    )

    candidate = RetrievalCandidate(
        memory=memory,
        lexical_score=0.5,
        semantic_score=0.8,
    )

    try:
        candidate.semantic_score = 0.2
        assert False
    except AttributeError:
        pass