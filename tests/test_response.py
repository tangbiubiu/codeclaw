"""
Response 响应类单元测试
"""

from closeclaw.provider.response import (
    ContentBlockType,
    FinishReason,
    Usage,
    TextContent,
    ImageContent,
    ToolCall,
    ToolResultContent,
    Message,
    Choice,
    ResponseMetadata,
    Response,
    StreamChunk,
    StreamResponse,
    convert_openai_response,
    convert_openai_stream_chunk,
)
from tests.conftest import create_mock_stream_chunk


class TestUsage:
    """Usage Token使用统计类测试"""

    def test_default_initialization(self):
        """测试默认初始化"""
        usage = Usage()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_initialization_with_values(self):
        """测试带值的初始化"""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.total_tokens == 150

    def test_to_dict(self):
        """测试转换为字典"""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        result = usage.to_dict()
        assert result == {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }


class TestTextContent:
    """TextContent 文本内容块测试"""

    def test_initialization(self):
        """测试初始化"""
        content = TextContent(text="Hello, world!")
        assert content.text == "Hello, world!"

    def test_to_dict(self):
        """测试转换为字典"""
        content = TextContent(text="Hello, world!")
        result = content.to_dict()
        assert result == {
            "type": "text",
            "text": "Hello, world!",
        }


class TestImageContent:
    """ImageContent 图像内容块测试"""

    def test_initialization_with_defaults(self):
        """测试默认参数初始化"""
        content = ImageContent(url="https://example.com/image.png")
        assert content.url == "https://example.com/image.png"
        assert content.detail == "auto"

    def test_initialization_with_custom_detail(self):
        """测试自定义detail初始化"""
        content = ImageContent(url="https://example.com/image.png", detail="high")
        assert content.url == "https://example.com/image.png"
        assert content.detail == "high"

    def test_to_dict(self):
        """测试转换为字典"""
        content = ImageContent(url="https://example.com/image.png", detail="low")
        result = content.to_dict()
        assert result == {
            "type": "image",
            "image_url": {
                "url": "https://example.com/image.png",
                "detail": "low",
            },
        }


class TestToolCall:
    """ToolCall 工具调用测试"""

    def test_initialization(self):
        """测试初始化"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Beijing"},
        )
        assert tool_call.id == "call_123"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"location": "Beijing"}

    def test_to_dict(self):
        """测试转换为字典"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Beijing"},
        )
        result = tool_call.to_dict()
        assert result == {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "Beijing"},
            },
        }

    def test_from_dict_with_json_string_arguments(self):
        """测试从字典创建（参数为JSON字符串）"""
        data = {
            "id": "call_456",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": '{"location": "Shanghai", "format": "celsius"}',
            },
        }
        tool_call = ToolCall.from_dict(data)
        assert tool_call.id == "call_456"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"location": "Shanghai", "format": "celsius"}

    def test_from_dict_with_dict_arguments(self):
        """测试从字典创建（参数为字典）"""
        data = {
            "id": "call_789",
            "type": "function",
            "function": {
                "name": "get_weather",
                "arguments": {"location": "Tokyo"},
            },
        }
        tool_call = ToolCall.from_dict(data)
        assert tool_call.id == "call_789"
        assert tool_call.name == "get_weather"
        assert tool_call.arguments == {"location": "Tokyo"}

    def test_from_dict_with_invalid_json(self):
        """测试从字典创建（无效JSON字符串）"""
        data = {
            "id": "call_invalid",
            "type": "function",
            "function": {
                "name": "test",
                "arguments": "not valid json",
            },
        }
        tool_call = ToolCall.from_dict(data)
        assert tool_call.arguments == {"raw": "not valid json"}


class TestToolResultContent:
    """ToolResultContent 工具执行结果测试"""

    def test_initialization(self):
        """测试初始化"""
        result = ToolResultContent(
            tool_call_id="call_123",
            output='{"temperature": 25}',
        )
        assert result.tool_call_id == "call_123"
        assert result.output == '{"temperature": 25}'
        assert result.is_error is False

    def test_initialization_with_error(self):
        """测试错误结果初始化"""
        result = ToolResultContent(
            tool_call_id="call_123",
            output="Error: Location not found",
            is_error=True,
        )
        assert result.is_error is True

    def test_to_dict(self):
        """测试转换为字典"""
        result = ToolResultContent(
            tool_call_id="call_123",
            output='{"temperature": 25}',
            is_error=False,
        )
        result_dict = result.to_dict()
        assert result_dict == {
            "type": "tool_result",
            "tool_call_id": "call_123",
            "output": '{"temperature": 25}',
            "is_error": False,
        }


class TestMessage:
    """Message 消息类测试"""

    def test_initialization_with_string_content(self):
        """测试字符串内容初始化"""
        message = Message(role="user", content="Hello!")
        assert message.role == "user"
        assert message.content == "Hello!"
        assert message.tool_calls is None

    def test_initialization_with_list_content(self):
        """测试列表内容初始化"""
        content = [TextContent(text="Hello!")]
        message = Message(role="assistant", content=content)
        assert message.role == "assistant"
        assert isinstance(message.content, list)
        assert message.tool_calls is None

    def test_initialization_with_tool_calls(self):
        """测试带工具调用的初始化"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Beijing"},
        )
        message = Message(
            role="assistant",
            content="",
            tool_calls=[tool_call],
        )
        assert message.tool_calls is not None
        assert len(message.tool_calls) == 1
        assert message.tool_calls[0].id == "call_123"

    def test_to_dict_with_string_content(self):
        """测试字符串内容转换为字典"""
        message = Message(role="user", content="Hello!")
        result = message.to_dict()
        assert result == {
            "role": "user",
            "content": "Hello!",
        }

    def test_to_dict_with_list_content(self):
        """测试列表内容转换为字典"""
        message = Message(role="assistant", content=[TextContent(text="Hi!")])
        result = message.to_dict()
        assert result["content"][0] == {"type": "text", "text": "Hi!"}

    def test_to_dict_with_tool_calls(self):
        """测试带工具调用的转换为字典"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Beijing"},
        )
        message = Message(role="assistant", content="", tool_calls=[tool_call])
        result = message.to_dict()
        assert "tool_calls" in result
        assert result["tool_calls"][0]["id"] == "call_123"


class TestChoice:
    """Choice 响应选项测试"""

    def test_initialization(self):
        """测试初始化"""
        message = Message(role="assistant", content="Hello!")
        choice = Choice(index=0, message=message, finish_reason=FinishReason.STOP)
        assert choice.index == 0
        assert choice.message == message
        assert choice.finish_reason == FinishReason.STOP

    def test_to_dict(self):
        """测试转换为字典"""
        message = Message(role="assistant", content="Hello!")
        choice = Choice(index=0, message=message, finish_reason=FinishReason.STOP)
        result = choice.to_dict()
        assert result == {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello!"},
            "finish_reason": "stop",
        }


class TestResponseMetadata:
    """ResponseMetadata 响应元数据测试"""

    def test_initialization(self):
        """测试初始化"""
        metadata = ResponseMetadata(model="gpt-4", id="chatcmpl-123")
        assert metadata.model == "gpt-4"
        assert metadata.id == "chatcmpl-123"

    def test_initialization_with_defaults(self):
        """测试默认初始化"""
        metadata = ResponseMetadata()
        assert metadata.model == ""
        assert metadata.id == ""

    def test_to_dict(self):
        """测试转换为字典"""
        metadata = ResponseMetadata(model="gpt-4", id="chatcmpl-123")
        result = metadata.to_dict()
        assert result == {"model": "gpt-4", "id": "chatcmpl-123"}


class TestResponse:
    """Response 完整响应类测试"""

    def test_initialization(self):
        """测试初始化"""
        response = Response()
        assert response.content == []
        assert response.metadata is not None
        assert response.usage is None
        assert response.choices == []
        assert response.error is None

    def test_initialization_full(self):
        """测试完整初始化"""
        metadata = ResponseMetadata(model="gpt-4", id="chatcmpl-123")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        content = [TextContent(text="Hello!")]
        message = Message(role="assistant", content=content)
        choice = Choice(index=0, message=message, finish_reason=FinishReason.STOP)

        response = Response(
            content=content,
            metadata=metadata,
            usage=usage,
            choices=[choice],
        )

        assert response.content[0].text == "Hello!"
        assert response.metadata.model == "gpt-4"
        assert response.usage.total_tokens == 15
        assert len(response.choices) == 1

    def test_get_text_content(self):
        """测试获取文本内容"""
        response = Response(
            content=[
                TextContent(text="Hello "),
                TextContent(text="World!"),
            ]
        )
        assert response.get_text_content() == "Hello \nWorld!"

    def test_get_text_content_with_empty_response(self):
        """测试获取空响应的文本内容"""
        response = Response(content=[])
        assert response.get_text_content() == ""

    def test_get_tool_calls(self):
        """测试获取工具调用"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Beijing"},
        )
        response = Response(
            content=[TextContent(text="Hello!"), tool_call],
        )
        tool_calls = response.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_123"

    def test_get_tool_calls_from_choices(self):
        """测试从choices获取工具调用"""
        tool_call = ToolCall(
            id="call_456",
            name="get_weather",
            arguments={"location": "Shanghai"},
        )
        message = Message(role="assistant", content="", tool_calls=[tool_call])
        choice = Choice(index=0, message=message)
        response = Response(choices=[choice])

        tool_calls = response.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_456"

    def test_is_error(self):
        """测试错误响应判断"""
        response_error = Response(error={"message": "API Error"})
        assert response_error.is_error is True

        response_ok = Response()
        assert response_ok.is_error is False

    def test_finish_reason(self):
        """测试结束原因"""
        message = Message(role="assistant", content="Hello!")
        choice = Choice(index=0, message=message, finish_reason=FinishReason.STOP)
        response = Response(choices=[choice])
        assert response.finish_reason == FinishReason.STOP

    def test_finish_reason_empty_choices(self):
        """测试空choices的结束原因"""
        response = Response(choices=[])
        assert response.finish_reason is None

    def test_to_dict(self):
        """测试转换为字典"""
        content = [TextContent(text="Hello!")]
        metadata = ResponseMetadata(model="gpt-4", id="chatcmpl-123")
        usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

        response = Response(
            content=content,
            metadata=metadata,
            usage=usage,
        )

        result = response.to_dict()
        assert "content" in result
        assert "metadata" in result
        assert "usage" in result
        assert result["usage"]["total_tokens"] == 15


class TestStreamChunk:
    """StreamChunk 流式响应块测试"""

    def test_initialization_with_string_delta(self):
        """测试字符串delta初始化"""
        chunk = StreamChunk(delta="Hello", index=0)
        assert chunk.delta == "Hello"
        assert chunk.index == 0

    def test_initialization_with_tool_call_delta(self):
        """测试工具调用delta初始化"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Beijing"},
        )
        chunk = StreamChunk(delta=tool_call, index=0)
        assert isinstance(chunk.delta, ToolCall)
        assert chunk.delta.id == "call_123"
        assert chunk.index == 0

    def test_initialization_with_finish_reason(self):
        """测试带结束原因的初始化"""
        chunk = StreamChunk(delta="Hello", index=0, finish_reason=FinishReason.STOP)
        assert chunk.delta == "Hello"
        assert chunk.finish_reason == FinishReason.STOP

    def test_is_final(self):
        """测试是否为最后一个块"""
        chunk_with_finish = StreamChunk(delta="Hello", finish_reason=FinishReason.STOP)
        assert chunk_with_finish.is_final is True

        chunk_without_finish = StreamChunk(delta="Hello")
        assert chunk_without_finish.is_final is False

    def test_to_dict_with_string_delta(self):
        """测试字符串delta转换为字典"""
        chunk = StreamChunk(delta="Hello", index=0)
        result = chunk.to_dict()
        assert result["delta"] == "Hello"
        assert result["index"] == 0

    def test_to_dict_with_tool_call_delta(self):
        """测试工具调用delta转换为字典"""
        tool_call = ToolCall(
            id="call_123",
            name="get_weather",
            arguments={"location": "Beijing"},
        )
        chunk = StreamChunk(delta=tool_call, index=0)
        result = chunk.to_dict()
        assert result["delta"]["id"] == "call_123"
        assert result["delta"]["function"]["name"] == "get_weather"


class TestStreamResponse:
    """StreamResponse 流式响应聚合器测试"""

    def test_initialization(self):
        """测试初始化"""
        aggregator = StreamResponse()
        assert aggregator.chunks == []

    def test_add_chunk(self):
        """测试添加块"""
        aggregator = StreamResponse()
        chunk = StreamChunk(delta=TextContent(text="Hello"), index=0)
        aggregator.add_chunk(chunk)
        assert len(aggregator.chunks) == 1

    def test_to_response_single_text_chunk(self):
        """测试单个文本块转换为响应"""
        aggregator = StreamResponse()
        chunk = StreamChunk(
            delta=TextContent(text="Hello, world!"),
            index=0,
            finish_reason=FinishReason.STOP,
        )
        aggregator.add_chunk(chunk)

        response = aggregator.to_response()
        assert response.get_text_content() == "Hello, world!"
        assert response.finish_reason == FinishReason.STOP

    def test_to_response_multiple_text_chunks(self):
        """测试多个文本块聚合"""
        aggregator = StreamResponse()
        aggregator.add_chunk(StreamChunk(delta=TextContent(text="Hello"), index=0))
        aggregator.add_chunk(StreamChunk(delta=TextContent(text=" "), index=1))
        aggregator.add_chunk(StreamChunk(delta=TextContent(text="World!"), index=2))
        aggregator.add_chunk(
            StreamChunk(delta=None, index=3, finish_reason=FinishReason.STOP)
        )

        response = aggregator.to_response()
        assert response.get_text_content() == "Hello World!"

    def test_to_response_with_string_deltas(self):
        """测试字符串delta聚合"""
        aggregator = StreamResponse()
        aggregator.add_chunk(StreamChunk(delta="Hello", index=0))
        aggregator.add_chunk(StreamChunk(delta=" ", index=1))
        aggregator.add_chunk(StreamChunk(delta="World!", index=2))
        aggregator.add_chunk(
            StreamChunk(delta=None, index=3, finish_reason=FinishReason.STOP)
        )

        response = aggregator.to_response()
        assert response.get_text_content() == "Hello World!"

    def test_to_response_with_metadata(self):
        """测试带元数据的聚合"""
        aggregator = StreamResponse()
        aggregator.add_chunk(
            StreamChunk(
                delta=TextContent(text="Hi"),
                index=0,
                metadata=ResponseMetadata(model="gpt-4", id="chatcmpl-123"),
            )
        )
        aggregator.add_chunk(
            StreamChunk(delta=None, index=1, finish_reason=FinishReason.STOP)
        )

        response = aggregator.to_response()
        assert response.metadata.model == "gpt-4"
        assert response.metadata.id == "chatcmpl-123"

    def test_get_text_content(self):
        """测试获取累积文本"""
        aggregator = StreamResponse()
        aggregator.add_chunk(StreamChunk(delta=TextContent(text="Line 1")))
        aggregator.add_chunk(StreamChunk(delta="Line 2"))
        aggregator.add_chunk(StreamChunk(delta=TextContent(text="Line 3")))

        assert aggregator.get_text_content() == "Line 1Line 2Line 3"


class TestConvertOpenAIResponse:
    """convert_openai_response 转换函数测试"""

    def test_convert_text_response(self, mock_openai_text_response_obj):
        """测试转换文本响应"""
        response = convert_openai_response(mock_openai_text_response_obj)

        assert isinstance(response, Response)
        assert response.metadata.model == "gpt-4o-mini"
        assert response.metadata.id == "chatcmpl-test123"
        assert response.usage is not None
        assert response.usage.prompt_tokens == 10
        assert response.usage.completion_tokens == 9
        assert response.usage.total_tokens == 19
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == FinishReason.STOP

    def test_convert_tool_call_response(self, mock_openai_tool_call_response_obj):
        """测试转换工具调用响应"""
        response = convert_openai_response(mock_openai_tool_call_response_obj)

        assert isinstance(response, Response)
        assert len(response.choices) == 1
        assert response.choices[0].finish_reason == FinishReason.TOOL_calls

        tool_calls = response.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].id == "call_abc123"
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"location": "Beijing"}

    def test_convert_empty_response(self):
        """测试转换空响应"""
        empty_response = {
            "id": "chatcmpl-empty",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        response = convert_openai_response(empty_response)
        assert isinstance(response, Response)
        assert response.choices == []


class TestConvertOpenAIStreamChunk:
    """convert_openai_stream_chunk 转换函数测试"""

    def test_convert_text_delta(self, mock_openai_stream_chunk_obj):
        """测试转换文本增量"""
        chunk = convert_openai_stream_chunk(mock_openai_stream_chunk_obj)

        assert isinstance(chunk, StreamChunk)
        assert chunk.delta == "Hello"
        assert chunk.index == 0
        assert chunk.metadata is not None
        assert chunk.metadata.id == "chatcmpl-test123"

    def test_convert_empty_delta(self):
        """测试转换空增量"""
        empty_chunk = create_mock_stream_chunk(
            {
                "id": "chatcmpl-empty",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": None,
                    }
                ],
            }
        )
        chunk = convert_openai_stream_chunk(empty_chunk)
        assert chunk.delta is None

    def test_convert_with_finish_reason(self):
        """测试转换带结束原因的块"""
        chunk_data = create_mock_stream_chunk(
            {
                "id": "chatcmpl-finish",
                "object": "chat.completion.chunk",
                "created": 1234567890,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        chunk = convert_openai_stream_chunk(chunk_data)
        assert chunk.finish_reason == FinishReason.STOP


class TestContentBlockType:
    """ContentBlockType 枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert ContentBlockType.TEXT.value == "text"
        assert ContentBlockType.IMAGE.value == "image"
        assert ContentBlockType.TOOL_USE.value == "tool_use"
        assert ContentBlockType.TOOL_RESULT.value == "tool_result"


class TestFinishReason:
    """FinishReason 枚举测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert FinishReason.STOP.value == "stop"
        assert FinishReason.LENGTH.value == "length"
        assert FinishReason.TOOL_calls.value == "tool_calls"
        assert FinishReason.CONTENT_FILTERED.value == "content_filtered"
        assert FinishReason.ERROR.value == "error"
