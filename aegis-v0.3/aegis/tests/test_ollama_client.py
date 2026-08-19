from core.llm_client import _from_ollama_response, _to_ollama_messages, _to_ollama_tool


def test_to_ollama_tool_converts_anthropic_schema():
    anthropic_tool = {
        "name": "read_file",
        "description": "Read a file.",
        "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}},
    }
    result = _to_ollama_tool(anthropic_tool)
    assert result["type"] == "function"
    assert result["function"]["name"] == "read_file"
    assert result["function"]["parameters"] == anthropic_tool["input_schema"]


def test_to_ollama_messages_passes_through_plain_string_turns():
    messages = [{"role": "user", "content": "hello"}]
    result = _to_ollama_messages(messages)
    assert result == [{"role": "user", "content": "hello"}]


def test_to_ollama_messages_converts_tool_use_and_tool_result():
    messages = [
        {"role": "user", "content": "what's my OS?"},
        {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": "t1", "name": "system_info", "input": {}}],
        },
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Linux"}],
        },
    ]
    result = _to_ollama_messages(messages)

    assert result[0] == {"role": "user", "content": "what's my OS?"}
    assert result[1]["role"] == "assistant"
    assert result[1]["tool_calls"][0]["function"]["name"] == "system_info"
    assert result[2] == {"role": "tool", "content": "Linux"}


def test_from_ollama_response_plain_text():
    data = {"message": {"role": "assistant", "content": "Hello there."}}
    response = _from_ollama_response(data)
    assert response.stop_reason == "end_turn"
    assert response.content[0].text == "Hello there."


def test_from_ollama_response_with_tool_call():
    data = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "system_info", "arguments": {}}}],
        }
    }
    response = _from_ollama_response(data)
    assert response.stop_reason == "tool_use"
    assert response.content[0].name == "system_info"
