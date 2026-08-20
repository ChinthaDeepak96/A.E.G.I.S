"""
A.E.G.I.S. Semantic Embeddings.

Provides a small abstraction around Sentence Transformers so the
rest of A.E.G.I.S. does not depend directly on the embedding model.

The embedding layer is responsible only for:

- loading the embedding model lazily
- generating normalized embeddings
- batch embedding
- cosine similarity

It does not know anything about SQLite or Memory objects.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"


@lru_cache(maxsize=4)
def _load_model(
    model_name: str,
) -> SentenceTransformer:
    """
    Load and cache a Sentence Transformer model.

    Models are loaded lazily so importing this module does not
    immediately download or initialize a model.
    """

    return SentenceTransformer(
        model_name
    )


class MemoryEmbedder:
    """
    Generate semantic embeddings for A.E.G.I.S. memory.

    Embeddings are normalized to unit length. This allows cosine
    similarity to be calculated efficiently using a dot product.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        self.model_name = model_name

    @property
    def dimension(self) -> int:
        """
        Return the embedding dimension.
        """

        model = _load_model(
            self.model_name
        )

        return int(
            model.get_embedding_dimension()
        )

    def encode(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Generate one normalized embedding.
        """

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "text must be a string"
            )

        text = text.strip()

        if not text:
            raise ValueError(
                "text cannot be empty"
            )

        model = _load_model(
            self.model_name
        )

        embedding = model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(
            embedding,
            dtype=np.float32,
        )

    def encode_many(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Generate normalized embeddings for multiple texts.
        """

        if not texts:
            return np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype=np.float32,
            )

        cleaned = [
            text.strip()
            for text in texts
        ]

        if any(
            not text
            for text in cleaned
        ):
            raise ValueError(
                "texts cannot contain empty strings"
            )

        model = _load_model(
            self.model_name
        )

        embeddings = model.encode(
            cleaned,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def similarity(
        self,
        first: str,
        second: str,
    ) -> float:
        """
        Calculate cosine similarity between two texts.

        Because both embeddings are normalized, cosine similarity
        is simply their dot product.
        """

        first_embedding = self.encode(
            first
        )

        second_embedding = self.encode(
            second
        )

        return self.vector_similarity(
            first_embedding,
            second_embedding,
        )

    @staticmethod
    def vector_similarity(
        first: np.ndarray,
        second: np.ndarray,
    ) -> float:
        """
        Calculate cosine similarity between two vectors.
        """

        first = np.asarray(
            first,
            dtype=np.float32,
        )

        second = np.asarray(
            second,
            dtype=np.float32,
        )

        if first.ndim != 1:
            raise ValueError(
                "first vector must be one-dimensional"
            )

        if second.ndim != 1:
            raise ValueError(
                "second vector must be one-dimensional"
            )

        if first.shape != second.shape:
            raise ValueError(
                "vectors must have the same dimensions"
            )

        first_norm = np.linalg.norm(
            first
        )

        second_norm = np.linalg.norm(
            second
        )

        if (
            first_norm == 0.0
            or second_norm == 0.0
        ):
            return 0.0

        similarity = (
            np.dot(
                first,
                second,
            )
            / (
                first_norm
                * second_norm
            )
        )

        return float(
            np.clip(
                similarity,
                -1.0,
                1.0,
            )
        )