"""
OpenAI 模型提供商
"""

from typing import Dict, Any, Iterator, AsyncIterator, cast
from openai import OpenAI, AsyncOpenAI, OpenAIError

from closeclaw.provider.base import ModelProvider, ProviderName
from closeclaw.provider.response import (
    Response,
    StreamChunk,
    convert_openai_response,
    convert_openai_stream_chunk,
    TextContent,

)
from closeclaw import constants as cs
from closeclaw.config import settings
from closeclaw.logger import app_logger as logger


class OpenAIProvider(ModelProvider):
    """
    OpenAI提供商
    """

    import openai

    __slots__ = (
        "api_key",
        "base_url",
    )

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        **configs: Dict[str, Any],
    ) -> None:
        """
        初始化OpenAI提供商

        优先使用传入的api_key和base_url，
        否则使用环境变量的OPENAI_API_KEY和OPENAI_API_BASE。
        """
        super().__init__(**configs)

        if api_key is None:
            api_key = cast(str | None, configs.get("api_key"))
            if api_key is None:
                raise ValueError("api_key is required")
        if base_url is None:
            if configs.get("base_url") is not None:
                base_url = cast(str | None, configs.get("base_url"))
            else:
                base_url = cs.OPENAI_DEFAULT_ENDPOINT

        self.api_key = api_key
        self.base_url = base_url

    @property
    def provider_name(self) -> str:
        """提供商名称"""
        return ProviderName.OPENAI.value

    def validate_model(self, model_id: str) -> bool:
        """
        验证连通性
        """
        try:
            model = self._create_model()
            model.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "This is a test message."}],
                temperature=0.5,
            )
        except OpenAIError as e:
            logger.error("OpenAI模型验证失败: %s", e)
            return False

        logger.success("OpenAI模型验证成功: %s", model_id)
        return True

    def _create_model(self) -> OpenAI:
        """
        创建模型客户端

        Returns:
            OpenAI客户端实例
        """
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.configs,
        )

    def _create_async_model(self) -> AsyncOpenAI:
        """
        创建异步模型客户端

        Returns:
            AsyncOpenAI客户端实例
        """
        return AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.configs,
        )

    def _create_error_response(
        self,
        message: str,
        model_id: str | None,
        original_error: Exception | None = None,
    ) -> Response:
        """
        创建错误响应

        Args:
            message: 错误消息
            model_id: 模型名称
            original_error: 原始异常

        Returns:
            错误响应对象
        """
        return Response(
            content=[TextContent(text="")],
            error={
                "message": message,
                "provider": self.provider_name,
                "model_id": model_id,
                "type": "provider_error",
            },
        )

    def invoke(
        self,
        messages: list[dict[str, Any]],
        model_id: str | None = None,
    ) -> Response:
        """
        同步非流式调用

        Args:
            messages: 消息列表
            model_id: 模型名称

        Returns:
            模型响应对象
        """
        if model_id is None:
            model_id = settings.CHAT_MODEL_NAME

        try:
            model = self._create_model()
            response = model.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.5,
            )
            return convert_openai_response(response)
        except OpenAIError as e:
            logger.error("OpenAI API 调用失败: %s", e)
            return self._create_error_response(
                message=f"OpenAI API 调用失败: {str(e)}",
                model_id=model_id,
                original_error=e,
            )

    async def ainvoke(
        self,
        messages: list[dict[str, Any]],
        model_id: str | None = None,
    ) -> Response:
        """
        异步非流式调用

        Args:
            messages: 消息列表
            model_id: 模型名称

        Returns:
            模型响应对象
        """
        if model_id is None:
            model_id = settings.CHAT_MODEL_NAME

        try:
            model = self._create_async_model()
            response = await model.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.5,
            )
            return convert_openai_response(response)
        except OpenAIError as e:
            logger.error("OpenAI API 调用失败: %s", e)
            return self._create_error_response(
                message=f"OpenAI API 调用失败: {str(e)}",
                model_id=model_id,
                original_error=e,
            )

    def stream(
        self,
        messages: list[dict[str, Any]],
        model_id: str | None = None,
    ) -> Iterator[StreamChunk]:
        """
        同步流式调用

        Args:
            messages: 消息列表
            model_id: 模型名称

        Returns:
            流式响应块迭代器
        """
        if model_id is None:
            model_id = settings.CHAT_MODEL_NAME

        try:
            model = self._create_model()
            response = model.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.5,
                stream=True,
            )
            for chunk in response:
                yield convert_openai_stream_chunk(chunk)
        except OpenAIError as e:
            logger.error("OpenAI API 流式调用失败: %s", e)
            error_response = self._create_error_response(
                message=f"OpenAI API 流式调用失败: {str(e)}",
                model_id=model_id,
                original_error=e,
            )
            yield StreamChunk(
                delta=TextContent(text=""),
                index=0,
                finish_reason=error_response.error,
            )

    async def astream(
        self,
        messages: list[dict[str, Any]],
        model_id: str | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """
        异步流式调用

        Args:
            messages: 消息列表
            model_id: 模型名称

        Returns:
            异步流式响应块迭代器
        """
        if model_id is None:
            model_id = settings.CHAT_MODEL_NAME

        try:
            model = self._create_async_model()
            response = await model.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=0.5,
                stream=True,
            )
            async for chunk in response:
                yield convert_openai_stream_chunk(chunk)
        except OpenAIError as e:
            logger.error("OpenAI API 流式调用失败: %s", e)
            error_response = self._create_error_response(
                message=f"OpenAI API 流式调用失败: {str(e)}",
                model_id=model_id,
                original_error=e,
            )
            yield StreamChunk(
                delta=TextContent(text=""),
                index=0,
                finish_reason=error_response.error,
            )
