"""
Additional tests for the LLM-backed memory extractor.

The extractor:
    conversation
        ↓
    LLM
        ↓
    JSON
        ↓
    MemoryCandidate

It must never directly persist memories.
"""

from core.llm_client import (
    LLMResponse,
    MockClient,
    TextBlock,
)

from memory.extractor import (
    MemoryCandidate,
    MemoryExtractor,
)
from memory.models import MemoryType


def make_client(payload: str) -> MockClient:
    return MockClient(
        responses=[
            LLMResponse(
                content=[
                    TextBlock(
                        text=payload
                    )
                ],
                stop_reason="end_turn",
            )
        ]
    )


def test_preference_candidate_is_parsed():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "User prefers VS Code.",
                    "type": "preference",
                    "importance": 0.8,
                    "confidence": 0.95,
                    "sensitivity": 0.0,
                    "tags": ["vscode", "preference"]
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "I prefer VS Code.",
            }
        ]
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert isinstance(
        candidate,
        MemoryCandidate,
    )

    assert (
        candidate.memory_type
        == MemoryType.PREFERENCE
    )

    assert (
        candidate.content
        == "User prefers VS Code."
    )

    assert candidate.importance == 0.8
    assert candidate.confidence == 0.95


def test_project_candidate_is_parsed():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "User is building A.E.G.I.S.",
                    "type": "episodic",
                    "importance": 0.9,
                    "confidence": 0.9,
                    "sensitivity": 0.0,
                    "tags": ["aegis", "project"]
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "I am building A.E.G.I.S.",
            }
        ]
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert (
        candidate.memory_type
        == MemoryType.EPISODIC
    )

    assert (
        "A.E.G.I.S."
        in candidate.content
    )

    assert "aegis" in candidate.tags
    assert "project" in candidate.tags


def test_semantic_candidate_is_parsed():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "A.E.G.I.S. uses Qwen 7B.",
                    "type": "semantic",
                    "importance": 0.9,
                    "confidence": 0.95,
                    "sensitivity": 0.0,
                    "tags": ["aegis", "qwen"]
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "A.E.G.I.S. uses Qwen 7B.",
            }
        ]
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    assert (
        candidate.memory_type
        == MemoryType.SEMANTIC
    )


def test_working_memory_candidate_is_parsed():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "A.E.G.I.S. is currently being tested.",
                    "type": "working",
                    "importance": 0.7,
                    "confidence": 0.85,
                    "sensitivity": 0.0,
                    "tags": ["testing"]
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "We are currently testing A.E.G.I.S.",
            }
        ]
    )

    assert len(candidates) == 1

    assert (
        candidates[0].memory_type
        == MemoryType.WORKING
    )


def test_question_can_produce_no_memory():
    client = make_client(
        '{"memories": []}'
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "What model should I use?",
            }
        ]
    )

    assert candidates == []


def test_greeting_can_produce_no_memory():
    client = make_client(
        '{"memories": []}'
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "Hello AEGIS!",
            }
        ]
    )

    assert candidates == []


def test_empty_message_list_returns_empty():
    client = make_client(
        '{"memories": []}'
    )

    extractor = MemoryExtractor(client)

    assert extractor.extract([]) == []


def test_blank_message_content_returns_empty():
    client = make_client(
        '{"memories": []}'
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "   ",
            }
        ]
    )

    assert candidates == []


def test_markdown_json_fence_is_supported():
    client = make_client(
        """
        ```json
        {
            "memories": [
                {
                    "content": "A.E.G.I.S. uses Gemma 4.",
                    "type": "semantic",
                    "importance": 0.9,
                    "confidence": 0.9,
                    "sensitivity": 0.0,
                    "tags": ["gemma"]
                }
            ]
        }
        ```
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "A.E.G.I.S. uses Gemma 4.",
            }
        ]
    )

    assert len(candidates) == 1

    assert (
        candidates[0].memory_type
        == MemoryType.SEMANTIC
    )


def test_invalid_memory_type_is_ignored():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "Invalid memory.",
                    "type": "not_a_real_type",
                    "importance": 0.8,
                    "confidence": 0.8,
                    "sensitivity": 0.0,
                    "tags": []
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "Invalid memory.",
            }
        ]
    )

    assert candidates == []


def test_candidate_has_automatic_extraction_source():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "A.E.G.I.S. uses Gemma 4.",
                    "type": "semantic",
                    "importance": 0.9,
                    "confidence": 0.9,
                    "sensitivity": 0.0,
                    "tags": []
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "A.E.G.I.S. uses Gemma 4.",
            }
        ]
    )

    assert len(candidates) == 1

    assert (
        candidates[0].source
        == "automatic_extraction"
    )


def test_candidate_contains_extraction_metadata():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "A.E.G.I.S. uses Gemma 4.",
                    "type": "semantic",
                    "importance": 0.9,
                    "confidence": 0.9,
                    "sensitivity": 0.0,
                    "tags": []
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "A.E.G.I.S. uses Gemma 4.",
            }
        ]
    )

    assert len(candidates) == 1

    assert (
        candidates[0].metadata[
            "extraction_method"
        ]
        == "llm"
    )


def test_extractor_does_not_create_database():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "A.E.G.I.S. uses Gemma 4.",
                    "type": "semantic",
                    "importance": 0.9,
                    "confidence": 0.9,
                    "sensitivity": 0.0,
                    "tags": []
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "A.E.G.I.S. uses Gemma 4.",
            }
        ]
    )

    assert len(candidates) == 1

    candidate = candidates[0]

    # A candidate has no persistent-memory ID.
    assert not hasattr(
        candidate,
        "id",
    )


def test_multiple_candidates_are_returned():
    client = make_client(
        """
        {
            "memories": [
                {
                    "content": "User prefers VS Code.",
                    "type": "preference",
                    "importance": 0.8,
                    "confidence": 0.9,
                    "sensitivity": 0.0,
                    "tags": ["vscode"]
                },
                {
                    "content": "A.E.G.I.S. uses Gemma 4.",
                    "type": "semantic",
                    "importance": 0.9,
                    "confidence": 0.95,
                    "sensitivity": 0.0,
                    "tags": ["aegis", "gemma"]
                }
            ]
        }
        """
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": (
                    "I prefer VS Code and A.E.G.I.S. "
                    "uses Gemma 4."
                ),
            }
        ]
    )

    assert len(candidates) == 2

    assert (
        candidates[0].memory_type
        == MemoryType.PREFERENCE
    )

    assert (
        candidates[1].memory_type
        == MemoryType.SEMANTIC
    )