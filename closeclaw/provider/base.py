from abc import ABC, abstractmethod
from typing import Dict, Any, AsyncIterator, Iterator, Union, cast
from enum import StrEnum
from openai import OpenAI, OpenAIError

from closeclaw import constants as cs
from closeclaw.config import settings
from closeclaw.logger import app_logger as logger
from closeclaw.provider.response import (
    Response,
    StreamResponse,
    StreamChunk,
    Usage,
    ContentBlock,
    ToolCall,
    TextContent,
)


class ProviderError(Exception):
    """模型提供商异常

    当模型API调用失败时抛出此异常。

    Attributes:
        message: 错误消息
        provider: 提供商名称
        model_id: 模型名称
        original_error: 原始异常
    """

    def __init__(
        self,
        message: str,
        provider: str = "unknown",
        model_id: str | None = None,
        original_error: Exception | None = None,
    ) -> None:
        self.message = message
        self.provider = provider
        self.model_id = model_id
        self.original_error = original_error
        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """转换为错误字典

        Returns:
            包含错误信息的字典
        """
        return {
            "error": {
                "message": self.message,
                "provider": self.provider,
                "model_id": self.model_id,
                "type": "provider_error",
            }
        }

    def __str__(self) -> str:
        return f"[{self.provider}:{self.model_id}] {self.message}"

# TODO: 当前只实现了OpenAI提供商
class ProviderName(StrEnum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    COHERE = "cohere"
    OPENAI_LIKE = "openai_like"
    VLLM = "vllm"

# TODO: invoke、ainvoke、stream、astream方法
class ModelProvider(ABC):
    """
    模型提供商基类
    """

    __slots__ = ("configs",)

    def __init__(self, **configs: Dict[str, Any]):
        self.configs = configs

    @abstractmethod
    def validate_model(self, model_id: str) -> bool:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """提供商名称"""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass

    @abstractmethod
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
        pass
