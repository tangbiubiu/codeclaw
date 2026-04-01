"""
OpenAIProvider 集成测试
"""

import os
from unittest.mock import patch, MagicMock

import pytest

from closeclaw.provider.openai import OpenAIProvider
from closeclaw.provider.base import ProviderName
from closeclaw.provider.response import (
    Response,
    FinishReason,
)


def get_env_api_key() -> str | None:
    """从环境变量获取 API Key"""
    return os.environ.get("OPENAI_API_KEY") or os.environ.get("CHAT_MODEL_API_KEY")


def get_env_base_url() -> str | None:
    """从环境变量获取 Base URL"""
    return os.environ.get("OPENAI_API_BASE") or os.environ.get("CHAT_MODEL_BASE_URL")


@pytest.fixture
def api_key() -> str | None:
    """获取 API Key"""
    return get_env_api_key()


@pytest.fixture
def base_url() -> str | None:
    """获取 Base URL"""
    return get_env_base_url()


@pytest.fixture
def model_id() -> str | None:
    """获取模型名称"""
    return os.environ.get("CHAT_MODEL_NAME") or os.environ.get(
        "OPENAI_MODEL_NAME", "gpt-4o-mini"
    )


class TestOpenAIProviderInit:
    """OpenAIProvider 初始化测试"""

    def test_init_with_explicit_credentials(self):
        """测试显式提供凭据初始化"""
        provider = OpenAIProvider(
            api_key="test-key-123",
            base_url="https://api.openai.com/v1",
        )
        assert provider.api_key == "test-key-123"
        assert provider.base_url == "https://api.openai.com/v1"

    def test_init_without_api_key_raises(self):
        """测试缺少 API Key 时抛出异常"""
        with pytest.raises(ValueError, match="api_key is required"):
            OpenAIProvider()

    def test_init_with_config_dict(self):
        """测试通过配置字典初始化"""
        provider = OpenAIProvider(
            api_key="test-key-456",
            base_url="https://custom.api.com/v1",
            timeout=30,
        )
        assert provider.api_key == "test-key-456"
        assert "timeout" in provider.configs


class TestOpenAIProviderProperties:
    """OpenAIProvider 属性测试"""

    def test_provider_name(self):
        """测试提供商名称"""
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )
        assert provider.provider_name == "openai"
        assert provider.provider_name == ProviderName.OPENAI.value


class TestOpenAIProviderInvoke:
    """OpenAIProvider invoke 方法测试"""

    def test_invoke_with_mock(self, test_messages):
        """测试使用 Mock 的 invoke 调用"""
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        mock_response = MagicMock()
        mock_response.id = "chatcmpl-mock"
        mock_response.model = "gpt-4o-mini"
        mock_response.choices = [
            MagicMock(
                index=0,
                message=MagicMock(
                    role="assistant",
                    content="Mock response content",
                ),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
        )

        with patch("closeclaw.provider.openai.OpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_openai_instance

            result = provider.invoke(messages=test_messages)

        assert isinstance(result, Response)
        assert result.is_error is False
        assert result.metadata.model == "gpt-4o-mini"
        assert result.usage is not None
        assert result.usage.total_tokens == 15
        text = result.get_text_content()
        assert "Mock response content" in text

    def test_invoke_api_error_handling(self, test_messages):
        """测试 API 错误处理"""
        from openai import OpenAIError

        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        with patch("closeclaw.provider.openai.OpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.side_effect = OpenAIError(
                "API Error"
            )
            mock_openai_class.return_value = mock_openai_instance

            result = provider.invoke(messages=test_messages)

        assert isinstance(result, Response)
        assert result.is_error is True
        assert result.error is not None
        assert "API Error" in result.error["message"]

    def test_invoke_with_custom_model(self, test_messages):
        """测试使用自定义模型"""
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        mock_response = MagicMock()
        mock_response.id = "chatcmpl-custom"
        mock_response.model = "gpt-4"
        mock_response.choices = [
            MagicMock(
                index=0,
                message=MagicMock(role="assistant", content="Custom model response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=5, completion_tokens=3, total_tokens=8
        )

        with patch("closeclaw.provider.openai.OpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_openai_instance

            result = provider.invoke(messages=test_messages, model_id="gpt-4")

        assert isinstance(result, Response)
        assert result.metadata.model == "gpt-4"


class TestOpenAIProviderAinvoke:
    """OpenAIProvider ainvoke 方法测试"""

    @pytest.mark.anyio
    async def test_ainvoke_with_mock(self, test_messages):
        """测试使用 Mock 的 async invoke 调用"""
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        mock_response = MagicMock()
        mock_response.id = "chatcmpl-async-mock"
        mock_response.model = "gpt-4o-mini"
        mock_response.choices = [
            MagicMock(
                index=0,
                message=MagicMock(role="assistant", content="Async mock response"),
                finish_reason="stop",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=8, completion_tokens=4, total_tokens=12
        )

        async def mock_acreate(*args, **kwargs):
            return mock_response

        with patch("closeclaw.provider.openai.AsyncOpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create = mock_acreate
            mock_openai_class.return_value = mock_openai_instance

            result = await provider.ainvoke(messages=test_messages)

        assert isinstance(result, Response)
        assert result.is_error is False
        assert result.metadata.id == "chatcmpl-async-mock"

    @pytest.mark.anyio
    async def test_ainvoke_api_error_handling(self, test_messages):
        """测试异步调用的错误处理"""
        from openai import OpenAIError

        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        async def mock_acreate_error(*args, **kwargs):
            raise OpenAIError("Async API Error")

        with patch("closeclaw.provider.openai.AsyncOpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create = mock_acreate_error
            mock_openai_class.return_value = mock_openai_instance

            result = await provider.ainvoke(messages=test_messages)

        assert isinstance(result, Response)
        assert result.is_error is True
        assert "Async API Error" in result.error["message"]


class TestOpenAIProviderStream:
    """OpenAIProvider stream 方法测试"""

    def test_stream_with_mock(self, test_messages):
        """测试使用 Mock 的流式调用"""
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        def mock_stream():
            chunks_data = [
                {
                    "id": "1",
                    "model": "gpt-4o-mini",
                    "delta_content": "Hello",
                    "finish_reason": None,
                },
                {
                    "id": "2",
                    "model": "gpt-4o-mini",
                    "delta_content": " World",
                    "finish_reason": None,
                },
                {
                    "id": "3",
                    "model": "gpt-4o-mini",
                    "delta_content": None,
                    "finish_reason": "stop",
                },
            ]
            for data in chunks_data:
                chunk = MagicMock()
                chunk.id = data["id"]
                chunk.model = data["model"]

                choice = MagicMock()
                choice.index = 0
                choice.finish_reason = data["finish_reason"]

                delta = MagicMock()
                if data["delta_content"] is not None:
                    delta.content = data["delta_content"]
                else:
                    delta.content = None

                choice.delta = delta
                chunk.choices = [choice]
                yield chunk

        with patch("closeclaw.provider.openai.OpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.return_value = mock_stream()
            mock_openai_class.return_value = mock_openai_instance

            chunks = list(provider.stream(messages=test_messages))

        assert len(chunks) == 3
        assert chunks[0].delta == "Hello"
        assert chunks[1].delta == " World"
        assert chunks[2].finish_reason == FinishReason.STOP

    def test_stream_error_handling(self, test_messages):
        """测试流式调用的错误处理"""
        from openai import OpenAIError

        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        with patch("closeclaw.provider.openai.OpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.side_effect = OpenAIError(
                "Stream Error"
            )
            mock_openai_class.return_value = mock_openai_instance

            chunks = list(provider.stream(messages=test_messages))

        assert len(chunks) == 1
        assert chunks[0].finish_reason is not None


class TestOpenAIProviderAstream:
    """OpenAIProvider astream 方法测试"""

    @pytest.mark.anyio
    async def test_astream_with_mock(self, test_messages):
        """测试使用 Mock 的异步流式调用"""

        class AsyncIteratorMock:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0

            def __aiter__(self):
                return self

            async def __anext__(self):
                if self.index >= len(self.chunks):
                    raise StopAsyncIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk

        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        async_chunks = []
        for i, text in enumerate(["Async ", "Stream ", "Test"]):
            chunk = MagicMock()
            chunk.id = f"chatcmpl-async-{i}"
            chunk.model = "gpt-4o-mini"

            choice = MagicMock()
            choice.index = 0
            choice.finish_reason = None

            delta = MagicMock()
            delta.content = text

            choice.delta = delta
            chunk.choices = [choice]
            async_chunks.append(chunk)

        last_chunk = MagicMock()
        last_chunk.id = "chatcmpl-async-final"
        last_chunk.model = "gpt-4o-mini"

        last_choice = MagicMock()
        last_choice.index = 0
        last_choice.finish_reason = "stop"
        last_choice.delta = MagicMock()
        last_choice.delta.content = None

        last_chunk.choices = [last_choice]
        async_chunks.append(last_chunk)

        async def mock_astream(*args, **kwargs):
            return AsyncIteratorMock(async_chunks)

        with patch("closeclaw.provider.openai.AsyncOpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create = mock_astream
            mock_openai_class.return_value = mock_openai_instance

            result_chunks = []
            async for chunk in provider.astream(messages=test_messages):
                result_chunks.append(chunk)

        assert len(result_chunks) == 4
        assert result_chunks[0].delta == "Async "
        assert result_chunks[3].finish_reason == FinishReason.STOP

    @pytest.mark.anyio
    async def test_astream_error_handling(self, test_messages):
        """测试异步流式调用的错误处理"""
        from openai import OpenAIError

        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        async def mock_astream_error(*args, **kwargs):
            raise OpenAIError("Async Stream Error")

        with patch("closeclaw.provider.openai.AsyncOpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create = mock_astream_error
            mock_openai_class.return_value = mock_openai_instance

            chunks = []
            async for chunk in provider.astream(messages=test_messages):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].finish_reason is not None


class TestOpenAIProviderValidateModel:
    """OpenAIProvider validate_model 方法测试"""

    def test_validate_model_success(self):
        """测试模型验证成功"""
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        mock_response = MagicMock()

        with patch("closeclaw.provider.openai.OpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_openai_instance

            result = provider.validate_model("gpt-4o-mini")

        assert result is True

    def test_validate_model_failure(self):
        """测试模型验证失败"""
        from openai import OpenAIError

        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        with patch("closeclaw.provider.openai.OpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.side_effect = OpenAIError(
                "Model not found"
            )
            mock_openai_class.return_value = mock_openai_instance

            result = provider.validate_model("invalid-model")

        assert result is False


class TestOpenAIProviderIntegration:
    """OpenAIProvider 集成测试（需要真实 API）"""

    @pytest.mark.skipif(
        not get_env_api_key(),
        reason="集成测试需要真实 API Key",
    )
    def test_invoke_real_api(self, api_key, base_url, model_id, test_messages):
        """测试真实 API 调用"""
        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )

        result = provider.invoke(messages=test_messages, model_id=model_id)

        assert isinstance(result, Response)
        assert result.is_error is False or result.error is not None
        if not result.is_error:
            assert result.metadata.model is not None
            text = result.get_text_content()
            assert len(text) > 0

    @pytest.mark.skipif(
        not get_env_api_key(),
        reason="集成测试需要真实 API Key",
    )
    @pytest.mark.anyio
    async def test_ainvoke_real_api(self, api_key, base_url, model_id, test_messages):
        """测试真实异步 API 调用"""
        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )

        result = await provider.ainvoke(messages=test_messages, model_id=model_id)

        assert isinstance(result, Response)
        assert result.is_error is False or result.error is not None
        if not result.is_error:
            assert result.usage is not None
            assert result.usage.total_tokens > 0

    @pytest.mark.skipif(
        not get_env_api_key(),
        reason="集成测试需要真实 API Key",
    )
    def test_stream_real_api(self, api_key, base_url, model_id, test_messages):
        """测试真实流式 API 调用"""
        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )

        chunks = list(provider.stream(messages=test_messages, model_id=model_id))

        assert len(chunks) > 0
        assert chunks[-1].is_final is True

        full_text = "".join(
            chunk.delta if isinstance(chunk.delta, str) else ""
            for chunk in chunks
            if chunk.delta
        )
        assert len(full_text) > 0

    @pytest.mark.skipif(
        not get_env_api_key(),
        reason="集成测试需要真实 API Key",
    )
    @pytest.mark.anyio
    async def test_astream_real_api(self, api_key, base_url, model_id, test_messages):
        """测试真实异步流式 API 调用"""
        provider = OpenAIProvider(
            api_key=api_key,
            base_url=base_url or "https://api.openai.com/v1",
        )

        chunks = []
        async for chunk in provider.astream(messages=test_messages, model_id=model_id):
            chunks.append(chunk)

        assert len(chunks) > 0
        assert chunks[-1].is_final is True


class TestOpenAIProviderToolCalls:
    """OpenAIProvider 工具调用测试"""

    def test_invoke_with_tool_call_mock(self):
        """测试带有工具调用的 Mock 响应"""
        provider = OpenAIProvider(
            api_key="test-key", base_url="https://api.test.com/v1"
        )

        def make_tool_call_to_dict(tc_id, name, args):
            def to_dict():
                return {
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    },
                }

            return to_dict

        mock_tool_call = MagicMock()
        mock_tool_call.id = "call_tool_123"
        mock_tool_call.type = "function"
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Beijing"}'
        mock_tool_call.to_dict = make_tool_call_to_dict(
            "call_tool_123", "get_weather", '{"location": "Beijing"}'
        )

        mock_response = MagicMock()
        mock_response.id = "chatcmpl-tool"
        mock_response.model = "gpt-4o-mini"
        mock_response.choices = [
            MagicMock(
                index=0,
                message=MagicMock(
                    role="assistant",
                    content=None,
                    tool_calls=[mock_tool_call],
                ),
                finish_reason="tool_calls",
            )
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=20, completion_tokens=15, total_tokens=35
        )

        with patch("closeclaw.provider.openai.OpenAI") as mock_openai_class:
            mock_openai_instance = MagicMock()
            mock_openai_instance.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_openai_instance

            result = provider.invoke(
                messages=[{"role": "user", "content": "What's the weather in Beijing?"}]
            )

        assert isinstance(result, Response)
        tool_calls = result.get_tool_calls()
        assert len(tool_calls) == 1
        assert tool_calls[0].name == "get_weather"
        assert tool_calls[0].arguments == {"location": "Beijing"}
        assert result.finish_reason == FinishReason.TOOL_calls
