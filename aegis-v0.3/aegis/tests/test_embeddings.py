from memory.embeddings import (
    DEFAULT_MODEL,
    MemoryEmbedder,
)


def test_default_model_name():
    embedder = MemoryEmbedder()

    assert (
        embedder.model_name
        == DEFAULT_MODEL
    )


def test_embedding_dimension():
    embedder = MemoryEmbedder()

    assert (
        embedder.dimension
        == 384
    )


def test_encode_returns_normalized_embedding():
    embedder = MemoryEmbedder()

    embedding = embedder.encode(
        "AEGIS uses Gemma 4."
    )

    assert embedding.shape == (
        384,
    )

    norm = float(
        (embedding ** 2).sum()
        ** 0.5
    )

    assert abs(
        norm - 1.0
    ) < 1e-4


def test_encode_many_returns_expected_shape():
    embedder = MemoryEmbedder()

    embeddings = embedder.encode_many(
        [
            "AEGIS uses Gemma.",
            "AEGIS uses Ollama.",
            "Python is useful.",
        ]
    )

    assert embeddings.shape == (
        3,
        384,
    )


def test_semantically_related_text_scores_higher():
    embedder = MemoryEmbedder()

    related = embedder.similarity(
        "AEGIS uses Gemma 4 through Ollama.",
        "What language model does AEGIS use?",
    )

    unrelated = embedder.similarity(
        "AEGIS uses Gemma 4 through Ollama.",
        "My favorite programming language is Python.",
    )

    assert related > unrelated


def test_vector_similarity_of_identical_vectors():
    embedder = MemoryEmbedder()

    embedding = embedder.encode(
        "AEGIS memory system."
    )

    similarity = (
        embedder.vector_similarity(
            embedding,
            embedding,
        )
    )

    assert abs(
        similarity - 1.0
    ) < 1e-5


def test_empty_text_is_rejected():
    embedder = MemoryEmbedder()

    try:
        embedder.encode("")
    except ValueError:
        pass
    else:
        raise AssertionError(
            "Empty text should raise ValueError"
        )


def test_empty_batch_returns_empty_array():
    embedder = MemoryEmbedder()

    embeddings = (
        embedder.encode_many([])
    )

    assert embeddings.shape == (
        0,
        384,
    )