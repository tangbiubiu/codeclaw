"""
pytest 配置文件
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest


def create_mock_openai_response(response_dict: dict) -> MagicMock:
    """从字典创建模拟的 OpenAI 响应对象"""
    mock = MagicMock()

    mock.id = response_dict.get("id", "")
    mock.model = response_dict.get("model", "")
    mock.object = response_dict.get("object", "")
    mock.created = response_dict.get("created", 0)

    mock.choices = []
    for choice_data in response_dict.get("choices", []):
        choice = MagicMock()
        choice.index = choice_data.get("index", 0)
        choice.finish_reason = choice_data.get("finish_reason")

        msg_data = choice_data.get("message", {})
        message = MagicMock()
        message.role = msg_data.get("role", "assistant")
        message.content = msg_data.get("content")

        tool_calls_data = msg_data.get("tool_calls")
        if tool_calls_data:
            message.tool_calls = []
            for tc_data in tool_calls_data:
                tc = MagicMock()
                tc.id = tc_data.get("id", "")
                tc.type = tc_data.get("type", "function")

                func_data = tc_data.get("function", {})
                tc.function = MagicMock()
                tc.function.name = func_data.get("name", "")
                tc.function.arguments = func_data.get("arguments", "{}")

                def make_to_dict(tc_data, tc):
                    def to_dict():
                        return {
                            "id": tc_data.get("id", ""),
                            "type": tc_data.get("type", "function"),
                            "function": {
                                "name": tc_data.get("function", {}).get("name", ""),
                                "arguments": tc_data.get("function", {}).get(
                                    "arguments", "{}"
                                ),
                            },
                        }

                    return to_dict

                tc.to_dict = make_to_dict(tc_data, tc)
                message.tool_calls.append(tc)
        else:
            message.tool_calls = None

        choice.message = message
        mock.choices.append(choice)

    usage_data = response_dict.get("usage", {})
    if usage_data:
        mock.usage = MagicMock()
        mock.usage.prompt_tokens = usage_data.get("prompt_tokens", 0)
        mock.usage.completion_tokens = usage_data.get("completion_tokens", 0)
        mock.usage.total_tokens = usage_data.get("total_tokens", 0)
    else:
        mock.usage = None

    return mock


@pytest.fixture(scope="session")
def env_file() -> Path:
    """获取 .env 文件路径"""
    return Path(__file__).parent.parent / ".env"


@pytest.fixture(scope="session")
def test_messages() -> list[dict[str, str]]:
    """通用测试消息列表"""
    return [
        {"role": "user", "content": "Hello, world!"},
    ]


@pytest.fixture(scope="session")
def multi_turn_messages() -> list[dict[str, str]]:
    """多轮对话消息列表"""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "2 + 2 equals 4."},
        {"role": "user", "content": "What about 3 + 3?"},
    ]


@pytest.fixture(scope="session")
def tool_call_messages() -> list[dict[str, str]]:
    """支持工具调用的消息列表"""
    return [
        {
            "role": "user",
            "content": "What is the weather in Beijing?",
        },
    ]


@pytest.fixture
def mock_openai_text_response() -> dict:
    """模拟 OpenAI 文本响应"""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help you today?",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 9,
            "total_tokens": 19,
        },
    }


@pytest.fixture
def mock_openai_text_response_obj(mock_openai_text_response) -> MagicMock:
    """模拟 OpenAI 文本响应对象"""
    return create_mock_openai_response(mock_openai_text_response)


@pytest.fixture
def mock_openai_stream_chunk() -> dict:
    """模拟 OpenAI 流式响应块"""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "content": "Hello",
                },
                "finish_reason": None,
            }
        ],
    }


def create_mock_stream_chunk(chunk_dict: dict) -> MagicMock:
    """从字典创建模拟的 OpenAI 流式块对象"""
    mock = MagicMock()

    mock.id = chunk_dict.get("id", "")
    mock.model = chunk_dict.get("model", "")
    mock.object = chunk_dict.get("object", "")
    mock.created = chunk_dict.get("created", 0)

    mock.choices = []
    for choice_data in chunk_dict.get("choices", []):
        choice = MagicMock()
        choice.index = choice_data.get("index", 0)
        choice.finish_reason = choice_data.get("finish_reason")

        delta_data = choice_data.get("delta", {})
        delta = MagicMock()
        delta.content = delta_data.get("content")

        tool_calls_data = delta_data.get("tool_calls")
        if tool_calls_data:
            delta.tool_calls = []
            for tc_data in tool_calls_data:
                tc = MagicMock()
                tc.id = tc_data.get("id", "")
                tc.type = tc_data.get("type", "function")

                func_data = tc_data.get("function", {})
                tc.function = MagicMock()
                tc.function.name = func_data.get("name", "")
                tc.function.arguments = func_data.get("arguments", "{}")

                delta.tool_calls.append(tc)
        else:
            delta.tool_calls = None

        choice.delta = delta
        mock.choices.append(choice)

    return mock


@pytest.fixture
def mock_openai_stream_chunk_obj(mock_openai_stream_chunk) -> MagicMock:
    """模拟 OpenAI 流式响应块对象"""
    return create_mock_stream_chunk(mock_openai_stream_chunk)


@pytest.fixture
def mock_openai_tool_call_response() -> dict:
    """模拟 OpenAI 工具调用响应"""
    return {
        "id": "chatcmpl-tool123",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Beijing"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 20,
            "total_tokens": 35,
        },
    }


@pytest.fixture
def mock_openai_tool_call_response_obj(mock_openai_tool_call_response) -> MagicMock:
    """模拟 OpenAI 工具调用响应对象"""
    return create_mock_openai_response(mock_openai_tool_call_response)
