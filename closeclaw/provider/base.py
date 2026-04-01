from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Dict, Any, AsyncIterator, Iterator

from closeclaw.provider.response import Response, StreamChunk


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
