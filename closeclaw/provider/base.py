from abc import ABC, abstractmethod
from typing import Dict, Any, cast
from enum import StrEnum
from openai import OpenAI, OpenAIError

from closeclaw import constants as cs


class ModelProvider(ABC):
    """
    模型提供商基类
    """

    __slots__ = ("configs",)

    def __init__(self, **configs: Dict[str, Any]):
        self.configs = configs

    @abstractmethod
    def create_model(self) -> Any:
        pass

    @abstractmethod
    def validate_model(self, model_id: str) -> bool:
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        pass


class Provider(StrEnum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GOOGLE = "google"
    AZURE = "azure"
    COHERE = "cohere"
    LOCAL = "local"
    VLLM = "vllm"


class OpenAIProvider(ModelProvider):
    """
    OpenAI提供商提供商
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
        return Provider.OPENAI.value

    def create_model(self) -> OpenAI:
        """
        创建模型
        """
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            **self.configs,
        )

    def validate_model(self, model_id: str) -> bool:
        """
        验证连通性
        """
        try:
            model = self.create_model()
            model.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": "This is a test message."}],
                temperature=0.5,
            )
        except OpenAIError:
            return False

        return True
