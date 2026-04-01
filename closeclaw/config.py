import json
import os
from typing import Any

import requests
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .logger import app_logger as logger


def download_with_requests(output_path: str = "cache/api.json") -> None:
    """下载 models.dev 项目的 API 配置文件

    从 models.dev 项目下载最新的 API 配置文件，用于获取模型信息。

    Args:
        output_path: 保存文件路径，默认 "cache/api.json"

    Raises:
        ConnectionError: 下载失败时抛出连接错误
        FileNotFoundError: 目录创建失败时抛出文件未找到错误

    Example:
        >>> download_with_requests("cache/api.json")
        >>> # 文件将保存到 cache/api.json
    """
    url = "https://models.dev/api.json"
    logger.info(f"Downloading API JSON from {url} to {output_path}")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        logger.warning(f"Failed to download API JSON from {url}: {e}")
        raise ConnectionError(f"Failed to download API JSON from {url}: {e}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)


def get_profile(
    provider: str, model_name: str, data_json: str = "cache/api.json"
) -> dict[str, Any]:
    """获取模型信息

    从 models.dev 项目的 API 配置文件中获取指定模型的配置信息。
    如果配置文件不存在，会自动下载最新的配置文件。

    Args:
        provider: 模型供应商，如 "openai"、"anthropic" 等
        model_name: 模型名称，如 "gpt-4"、"claude-3" 等
        data_json: 模型配置文件路径，默认 "cache/api.json"

    Returns:
        dict[str, Any]: 包含模型配置信息的字典

    Raises:
        ValueError: 当指定的供应商或模型不存在时抛出
        ConnectionError: 下载配置文件失败时抛出

    Example:
        >>> profile = get_profile("openai", "gpt-4")
        >>> print(profile["max_tokens"])
    """
    try:
        download_with_requests()
        logger.info(f"成功更新 API JSON 文件: {data_json}")
    except ConnectionError:
        logger.error(f"Failed to download API JSON file: {data_json}")

    with open(data_json) as f:
        data = json.load(f)

    try:
        result = data[provider]["models"][model_name]
        result["model_provider"] = provider
    except KeyError:
        logger.warning(f"Provider {provider} or model {model_name} not found")
        raise ValueError(f"Provider {provider} or model {model_name} not found")
    return result


class ContextLimit(BaseModel):
    """上下文限制配置

    定义模型处理文本时的上下文长度限制。

    Attributes:
        context: 上下文最大token数，必须大于0
        output: 输出最大token数，必须大于0
    """

    context: int = Field(gt=0, description="上下文最大token数")
    output: int = Field(gt=0, description="输出最大token数")


class ModelCost(BaseModel):
    """模型成本配置

    定义模型使用过程中的成本计算参数。

    Attributes:
        cache_read: 缓存读取成本，必须大于等于0
        input: 输入token成本，必须大于等于0
        output: 输出token成本，必须大于等于0
    """

    cache_read: float = Field(ge=0, description="缓存读取成本")
    input: float = Field(ge=0, description="输入token成本")
    output: float = Field(ge=0, description="输出token成本")


class ChatModelConfig(BaseModel):
    """聊天模型配置

    定义聊天模型的相关配置参数，用于与AI模型进行对话交互。

    Attributes:
        model_provider: 模型供应商，如 "openai"、"anthropic" 等
        model: 模型名称，如 "gpt-4"、"claude-3" 等
        api_key: API密钥，用于身份验证
        base_url: API基础URL，用于自定义API端点
        max_tokens: 最大token数，控制生成文本长度
        temperature: 温度参数，控制生成文本的随机性
        limit: 上下文限制配置
        reasoning: 是否支持推理功能
        tool_call: 是否支持工具调用功能
        cost: 成本配置参数
    """

    model_provider: str = Field(description="模型供应商")
    model: str = Field(description="模型名称")
    api_key: str | None = Field(None, description="API密钥")
    base_url: str | None = Field(None, description="API基础URL")
    max_tokens: int | None = Field(None, gt=0, description="最大token数")
    temperature: float | None = Field(None, ge=0, le=2, description="温度参数")
    limit: ContextLimit | None = Field(None, description="上下文限制")
    reasoning: bool | None = Field(None, description="是否支持推理")
    tool_call: bool | None = Field(None, description="是否支持工具调用")
    cost: ModelCost | None = Field(None, description="成本配置")


class AppConfig(BaseSettings):
    """应用主配置类

    从 .env 文件和环境变量加载配置。

    Attributes:
        PROJECT_ENV: 项目环境，如 "dev"、"prod" 等
        CHAT_API_TYPE: 聊天模型API类型
        CHAT_MODEL_PROVIDER: 聊天模型供应商
        CHAT_MODEL_NAME: 聊天模型名称
        CHAT_MODEL_API_KEY: 聊天模型API密钥
        CHAT_MODEL_BASE_URL: 聊天模型基础URL
        MAX_TOKENS: 最大token数
        TEMPERATURE: 温度参数
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="_",
        case_sensitive=False,
        extra="ignore",
    )

    # 聊天模型配置
    CHAT_API_TYPE: str = Field("openai", description="聊天模型API类型")
    CHAT_MODEL_PROVIDER: str = Field(None, description="聊天模型供应商")
    CHAT_MODEL_NAME: str = Field(None, description="聊天模型名称")
    CHAT_MODEL_API_KEY: str | None = Field(None, description="聊天模型API密钥")
    CHAT_MODEL_BASE_URL: str | None = Field(None, description="聊天模型基础URL")
    MAX_TOKENS: int = Field(4096, gt=0, description="最大token数")
    TEMPERATURE: float = Field(0.7, ge=0, le=2, description="温度参数")

    @property
    def chat_model_config(self) -> ChatModelConfig:
        """获取聊天模型配置"""
        if self.CHAT_MODEL_PROVIDER != "local":
            try:
                profile = get_profile(self.CHAT_MODEL_PROVIDER, self.CHAT_MODEL_NAME)
            except (ValueError, ConnectionError, FileNotFoundError):
                profile = {}
            config_data = {
                "api_type": self.CHAT_API_TYPE,
                "model_provider": self.CHAT_MODEL_PROVIDER,
                "model": self.CHAT_MODEL_NAME,
                "api_key": self.CHAT_MODEL_API_KEY,
                "base_url": self.CHAT_MODEL_BASE_URL,
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.TEMPERATURE,
            }

            config_data.update(profile)

        else:
            profile = {}
            config_data = {
                "api_type": self.CHAT_API_TYPE,
                "model_provider": self.CHAT_API_TYPE,
                "model": self.CHAT_MODEL_NAME,
                "api_key": self.CHAT_MODEL_API_KEY,
                "base_url": self.CHAT_MODEL_BASE_URL,
                "max_tokens": self.MAX_TOKENS,
                "temperature": self.TEMPERATURE,
            }

        return ChatModelConfig(**config_data)

    def validate_config(self) -> bool:
        """验证配置是否完整有效"""
        try:
            if self.CHAT_MODEL_BASE_URL and not self.CHAT_MODEL_BASE_URL.startswith(
                ("http://", "https://")
            ):
                raise ValueError("聊天模型基础URL格式不正确")

            return True
        except ValueError as e:
            print(f"配置验证失败: {e}")
            return False

    def log_config(self) -> None:
        """记录当前完整配置信息"""
        logger.info("==================== 模型配置 ====================")
        logger.info(f"项目环境: {self.PROJECT_ENV}")
        logger.info(f"聊天模型API类型: {self.CHAT_API_TYPE}")
        logger.info(f"聊天模型供应商: {self.CHAT_MODEL_PROVIDER}")
        logger.info(f"聊天模型名称: {self.CHAT_MODEL_NAME}")
        logger.info(f"聊天模型基础URL: {self.CHAT_MODEL_BASE_URL}")
        logger.info(f"最大token数: {self.MAX_TOKENS}")
        logger.info(f"温度参数: {self.TEMPERATURE}")


settings = AppConfig()
