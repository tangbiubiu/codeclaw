"""
OpenAI 模型提供商
"""

from typing import Dict, Any, Iterator, AsyncIterator, cast
from openai import OpenAI, AsyncOpenAI, OpenAIError

from codeclaw.provider.base import ModelProvider, ProviderName
from codeclaw.provider.response import (
    Response,
    StreamChunk,
    FinishReason,
    convert_openai_response,
    convert_openai_stream_chunk,
    TextContent,
)
from codeclaw import constants as cs
from codeclaw.config import settings
from codeclaw.logger import app_logger as logger


class OpenAIProvider(ModelProvider):
    """
    OpenAI提供商
    """

    import openai

    __slots__ = (
        "api_key",
        "base_url",
        "_sync_client",
        "_async_client",
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
        self._sync_client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None

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
            model.models.list()
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
        if self._sync_client is None:
            self._sync_client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **self.configs,
            )
        return self._sync_client

    def _create_async_model(self) -> AsyncOpenAI:
        """
        创建异步模型客户端

        Returns:
            AsyncOpenAI客户端实例
        """
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                **self.configs,
            )
        return self._async_client

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
                temperature=settings.TEMPERATURE,
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
                temperature=settings.TEMPERATURE,
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
                temperature=settings.TEMPERATURE,
                stream=True,
            )
            for chunk in response:
                yield convert_openai_stream_chunk(chunk)
        except OpenAIError as e:
            logger.error("OpenAI API 流式调用失败: %s", e)
            yield StreamChunk(
                delta=TextContent(text=""),
                index=0,
                finish_reason=FinishReason.STOP,
            )

    async def astream(  # type: ignore[override]
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
                temperature=settings.TEMPERATURE,
                stream=True,
            )
            async for chunk in response:
                yield convert_openai_stream_chunk(chunk)
        except OpenAIError as e:
            logger.error("OpenAI API 流式调用失败: %s", e)
            yield StreamChunk(
                delta=TextContent(text=""),
                index=0,
                finish_reason=FinishReason.STOP,
            )
