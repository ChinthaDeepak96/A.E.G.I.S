from core.llm_client import (
    LLMResponse,
    MockClient,
    TextBlock,
)

from memory.extractor import MemoryExtractor
from memory.models import MemoryType


def test_extractor_parses_valid_json():
    client = MockClient(
        responses=[
            LLMResponse(
                content=[
                    TextBlock(
                        text="""{
                            "memories": [
                                {
                                    "content": "A.E.G.I.S. uses Gemma 4.",
                                    "type": "semantic",
                                    "importance": 0.9,
                                    "confidence": 0.95,
                                    "sensitivity": 0.0,
                                    "tags": ["aegis", "gemma"]
                                }
                            ]
                        }"""
                    )
                ],
                stop_reason="end_turn",
            )
        ]
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

    memory = candidates[0]

    assert memory.content == "A.E.G.I.S. uses Gemma 4."
    assert memory.memory_type == MemoryType.SEMANTIC
    assert memory.importance == 0.9
    assert memory.confidence == 0.95


def test_extractor_handles_empty_result():
    client = MockClient(
        responses=[
            LLMResponse(
                content=[
                    TextBlock(
                        text='{"memories": []}'
                    )
                ],
                stop_reason="end_turn",
            )
        ]
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "Hello.",
            }
        ]
    )

    assert candidates == []


def test_extractor_handles_invalid_json():
    client = MockClient(
        responses=[
            LLMResponse(
                content=[
                    TextBlock(
                        text="This is not JSON."
                    )
                ],
                stop_reason="end_turn",
            )
        ]
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [
            {
                "role": "user",
                "content": "Hello.",
            }
        ]
    )

    assert candidates == []


def test_extractor_clamps_scores():
    client = MockClient(
        responses=[
            LLMResponse(
                content=[
                    TextBlock(
                        text="""{
                            "memories": [
                                {
                                    "content": "Test",
                                    "type": "semantic",
                                    "importance": 4,
                                    "confidence": -2,
                                    "sensitivity": 9,
                                    "tags": []
                                }
                            ]
                        }"""
                    )
                ],
                stop_reason="end_turn",
            )
        ]
    )

    extractor = MemoryExtractor(client)

    candidates = extractor.extract(
        [{"role": "user", "content": "Test"}]
    )

    assert len(candidates) == 1

    memory = candidates[0]

    assert memory.importance == 1.0
    assert memory.confidence == 0.0
    assert memory.sensitivity == 1.0