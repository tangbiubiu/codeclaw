"""
1. 任务规划
2. 工具调用
3. 多轮对话
4.
"""

from codeclaw.config import settings
from codeclaw.logger import app_logger as logger
from codeclaw.provider.base import ModelProvider


class AgentLoop:
    """
    智能体循环类
    """
    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider

    def validate_model(self, model_id: str) -> bool:
        """
        验证模型是否存在

        Args:
            model_id: 模型名称

        Returns:
            是否存在
        """
        return self.model_provider.validate_model(model_id)
