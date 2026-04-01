from abc import ABC, abstractmethod
from typing import Dict, Any


class Tool(ABC):
    """
    工具基类
    """

    timeout: int | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """工具参数，json schema"""
        pass

    @abstractmethod
    async def exec(self, **params) -> str:
        """工具执行"""
        pass

    def openai_schema(self) -> Dict[str, Any]:
        """生成openai tools参数"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
