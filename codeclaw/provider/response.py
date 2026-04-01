"""
模型响应接口定义

包含工具调用、文本内容、token使用统计等响应相关的数据结构
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Dict, List, Optional, Union


class ContentBlockType(StrEnum):
    """内容块类型"""

    TEXT = "text"
    IMAGE = "image"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"


class FinishReason(StrEnum):
    """结束原因"""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALLS = "tool_calls"
    CONTENT_FILTERED = "content_filtered"
    ERROR = "error"


@dataclass
class Usage:
    """
    Token使用统计

    Attributes:
        prompt_tokens: 输入提示的token数量
        completion_tokens: 生成内容的token数量
        total_tokens: 总token数量
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class TextContent:
    """
    文本内容块

    Attributes:
        text: 文本内容
    """

    text: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": ContentBlockType.TEXT.value,
            "text": self.text,
        }


@dataclass
class ImageContent:
    """
    图像内容块

    Attributes:
        url: 图像URL
        detail: 图像详情级别 ("low", "high", "auto")
    """

    url: str
    detail: str = "auto"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": ContentBlockType.IMAGE.value,
            "image_url": {
                "url": self.url,
                "detail": self.detail,
            },
        }


@dataclass
class ToolCallArgument:
    """
    工具调用参数

    Attributes:
        name: 参数名称
        value: 参数值 (JSON字符串)
    """

    name: str
    value: str

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import json

        try:
            return json.loads(self.value)
        except json.JSONDecodeError:
            return {"raw": self.value}


@dataclass
class ToolCall:
    """
    工具调用

    Attributes:
        id: 工具调用ID
        name: 工具名称
        arguments: 工具参数 (字典形式)
    """

    id: str
    name: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": self.arguments,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ToolCall":
        """从字典创建工具调用"""
        func = data.get("function", {})
        args_str = func.get("arguments", "{}")
        if isinstance(args_str, str):
            import json

            try:
                args = json.loads(args_str)
            except json.JSONDecodeError:
                args = {"raw": args_str}
        else:
            args = args_str

        return cls(
            id=data.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
        )


@dataclass
class ToolResultContent:
    """
    工具执行结果内容块

    Attributes:
        tool_call_id: 对应的工具调用ID
        output: 工具执行结果
        is_error: 是否为错误结果
    """

    tool_call_id: str
    output: str
    is_error: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "type": ContentBlockType.TOOL_RESULT.value,
            "tool_call_id": self.tool_call_id,
            "output": self.output,
            "is_error": self.is_error,
        }


ContentBlock = Union[TextContent, ImageContent, ToolCall, ToolResultContent]
"""内容块联合类型"""


@dataclass
class Choice:
    """
    响应选项

    Attributes:
        index: 选项索引
        message: 响应的消息内容
        finish_reason: 结束原因
    """

    index: int = 0
    message: Optional["Message"] = None
    finish_reason: Optional[FinishReason] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "index": self.index,
            "message": self.message.to_dict() if self.message else None,
            "finish_reason": self.finish_reason.value if self.finish_reason else None,
        }


@dataclass
class Message:
    """
    消息

    Attributes:
        role: 角色 ("user", "assistant", "system", "tool")
        content: 消息内容 (可以是字符串或内容块列表)
        tool_calls: 工具调用列表
    """

    role: str
    content: Union[str, List[ContentBlock]] = ""
    tool_calls: Optional[List[ToolCall]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result: Dict[str, Any] = {
            "role": self.role,
            "content": self.content
            if isinstance(self.content, str)
            else [
                block.to_dict() if hasattr(block, "to_dict") else block
                for block in self.content
            ],
        }

        if self.tool_calls:
            result["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]

        return result


@dataclass
class ResponseMetadata:
    """
    响应元数据

    Attributes:
        model: 实际使用的模型名称
        id: 响应ID
    """

    model: str = ""
    id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "model": self.model,
            "id": self.id,
        }


@dataclass
class Response:
    """
    模型响应

    统一封装各种模型返回结果，包括文本内容、工具调用、token使用等信息。

    Attributes:
        content: 响应内容列表
        metadata: 响应元数据
        usage: Token使用统计
        choices: 响应选项列表
        error: 错误信息
    """

    content: List[ContentBlock] = field(default_factory=list)
    metadata: ResponseMetadata = field(default_factory=ResponseMetadata)
    usage: Optional[Usage] = None
    choices: List[Choice] = field(default_factory=list)
    error: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result: Dict[str, Any] = {
            "content": [
                block.to_dict() if hasattr(block, "to_dict") else block
                for block in self.content
            ],
            "metadata": self.metadata.to_dict(),
        }

        if self.usage:
            result["usage"] = self.usage.to_dict()

        if self.choices:
            result["choices"] = [choice.to_dict() for choice in self.choices]

        if self.error:
            result["error"] = self.error

        return result

    def get_text_content(self) -> str:
        """
        获取文本内容

        Returns:
            合并后的文本内容
        """
        text_parts = []
        for block in self.content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, str):
                text_parts.append(block)
        return "\n".join(text_parts)

    def get_tool_calls(self) -> List[ToolCall]:
        """
        获取所有工具调用

        Returns:
            工具调用列表
        """
        tool_calls = []
        for block in self.content:
            if isinstance(block, ToolCall):
                tool_calls.append(block)
        if self.choices:
            for choice in self.choices:
                if choice.message and choice.message.tool_calls:
                    tool_calls.extend(choice.message.tool_calls)
        return tool_calls

    @property
    def is_error(self) -> bool:
        """是否为错误响应"""
        return self.error is not None

    @property
    def finish_reason(self) -> Optional[FinishReason]:
        """获取结束原因"""
        if self.choices and self.choices[0].finish_reason:
            return self.choices[0].finish_reason
        return None


@dataclass
class StreamChunk:
    """
    流式响应块

    Attributes:
        delta: 增量内容 (字符串形式的文本或 ToolCall 对象)
        index: 块索引
        finish_reason: 结束原因
        usage: 增量token使用 (部分provider支持)
    """

    delta: Union[TextContent, ToolCall, str, None] = None
    index: int = 0
    finish_reason: Optional[FinishReason] = None
    usage: Optional[Usage] = None
    metadata: Optional[ResponseMetadata] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        delta_dict: Any = None
        if isinstance(self.delta, str):
            delta_dict = self.delta
        elif isinstance(self.delta, TextContent):
            delta_dict = self.delta.to_dict()
        elif isinstance(self.delta, ToolCall):
            delta_dict = self.delta.to_dict()

        result: Dict[str, Any] = {
            "delta": delta_dict,
            "index": self.index,
        }

        if self.finish_reason:
            result["finish_reason"] = self.finish_reason.value

        if self.usage:
            result["usage"] = self.usage.to_dict()

        if self.metadata:
            result["metadata"] = self.metadata.to_dict()

        return result

    @property
    def is_final(self) -> bool:
        """是否为最后一个块"""
        return self.finish_reason is not None


@dataclass
class StreamResponse:
    """
    流式响应聚合器

    用于聚合流式响应块，生成完整的响应

    Attributes:
        chunks: 接收的块列表
    """

    chunks: List[StreamChunk] = field(default_factory=list)

    def add_chunk(self, chunk: StreamChunk) -> None:
        """添加流式块"""
        self.chunks.append(chunk)

    def to_response(self) -> Response:
        """
        转换为完整响应

        Returns:
            聚合后的完整响应
        """
        content: List[ContentBlock] = []
        tool_calls: Dict[str, ToolCall] = {}
        text_parts: List[str] = []
        final_metadata = ResponseMetadata()
        final_usage: Optional[Usage] = None
        final_finish_reason: Optional[FinishReason] = None

        for chunk in self.chunks:
            if chunk.metadata:
                final_metadata = chunk.metadata

            if chunk.usage:
                final_usage = chunk.usage

            if chunk.finish_reason:
                final_finish_reason = chunk.finish_reason

            if chunk.delta is None:
                continue

            if isinstance(chunk.delta, str):
                text_parts.append(chunk.delta)
            elif isinstance(chunk.delta, TextContent):
                text_parts.append(chunk.delta.text)
            elif isinstance(chunk.delta, ToolCall):
                if chunk.delta.id:
                    tool_calls[chunk.delta.id] = chunk.delta
                else:
                    content.append(chunk.delta)

        if text_parts:
            content.append(TextContent(text="".join(text_parts)))

        if tool_calls:
            content.extend(tool_calls.values())

        choices = []
        if content or final_finish_reason:
            message = Message(role="assistant", content=content if content else "")
            if tool_calls:
                message.tool_calls = list(tool_calls.values())
            choices.append(
                Choice(
                    index=0,
                    message=message,
                    finish_reason=final_finish_reason,
                )
            )

        return Response(
            content=content,
            metadata=final_metadata,
            usage=final_usage,
            choices=choices,
        )

    def get_text_content(self) -> str:
        """
        获取累积的文本内容

        Returns:
            累积的文本内容
        """
        text_parts = []
        for chunk in self.chunks:
            if chunk.delta is None:
                continue
            if isinstance(chunk.delta, str):
                text_parts.append(chunk.delta)
            elif isinstance(chunk.delta, TextContent):
                text_parts.append(chunk.delta.text)
            elif isinstance(chunk.delta, ToolCall):
                continue
        return "".join(text_parts)


def convert_openai_response(response: Any) -> Response:
    """
    将OpenAI响应转换为统一格式

    Args:
        response: OpenAI API响应对象

    Returns:
        统一格式的Response对象
    """
    content: List[ContentBlock] = []
    tool_calls: Dict[str, ToolCall] = {}
    choices: List[Choice] = []

    if hasattr(response, "choices") and response.choices:
        for idx, choice in enumerate(response.choices):
            message = Message(role="assistant", content="")

            if hasattr(choice, "message"):
                msg = choice.message

                if hasattr(msg, "content") and msg.content:
                    content.append(TextContent(text=msg.content))

                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_call = ToolCall.from_dict(
                            tc.to_dict() if hasattr(tc, "to_dict") else tc
                        )
                        tool_calls[tool_call.id] = tool_call

                if hasattr(msg, "role"):
                    message.role = msg.role

            message.content = content if content else ""
            if tool_calls:
                message.tool_calls = list(tool_calls.values())

            finish_reason = None
            if hasattr(choice, "finish_reason") and choice.finish_reason:
                try:
                    finish_reason = FinishReason(choice.finish_reason)
                except ValueError:
                    finish_reason = FinishReason.STOP

            choices.append(
                Choice(
                    index=idx,
                    message=message,
                    finish_reason=finish_reason,
                )
            )

    metadata = ResponseMetadata(
        model=getattr(response, "model", ""),
        id=getattr(response, "id", ""),
    )

    usage = None
    if hasattr(response, "usage") and response.usage:
        usage = Usage(
            prompt_tokens=getattr(response.usage, "prompt_tokens", 0),
            completion_tokens=getattr(response.usage, "completion_tokens", 0),
            total_tokens=getattr(response.usage, "total_tokens", 0),
        )

    return Response(
        content=content if content else [TextContent(text="")],
        metadata=metadata,
        usage=usage,
        choices=choices,
    )


def convert_openai_stream_chunk(chunk: Any) -> StreamChunk:
    """
    将OpenAI流式块转换为统一格式

    Args:
        chunk: OpenAI流式事件对象

    Returns:
        统一格式的StreamChunk对象
    """
    delta: Union[TextContent, ToolCall, str, None] = None
    finish_reason: Optional[FinishReason] = None
    index = 0
    usage: Optional[Usage] = None
    metadata: Optional[ResponseMetadata] = None

    if hasattr(chunk, "choices") and chunk.choices:
        choice = chunk.choices[0]
        index = getattr(choice, "index", 0)

        if hasattr(choice, "finish_reason") and choice.finish_reason:
            try:
                finish_reason = FinishReason(choice.finish_reason)
            except ValueError:
                finish_reason = FinishReason.STOP

        if hasattr(choice, "delta") and choice.delta:
            delta_obj = choice.delta

            if hasattr(delta_obj, "content") and delta_obj.content:
                delta = str(delta_obj.content)

            elif hasattr(delta_obj, "tool_calls") and delta_obj.tool_calls:
                tc_data = delta_obj.tool_calls[0]
                if hasattr(tc_data, "to_dict"):
                    tc_data = tc_data.to_dict()
                tool_call = ToolCall.from_dict(tc_data)
                delta = tool_call

        if hasattr(choice, "usage") and choice.usage:
            usage = Usage(
                prompt_tokens=getattr(choice.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(choice.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(choice.usage, "total_tokens", 0) or 0,
            )

    if hasattr(chunk, "model"):
        metadata = ResponseMetadata(model=chunk.model)

    if hasattr(chunk, "id"):
        if metadata is None:
            metadata = ResponseMetadata()
        metadata.id = chunk.id

    return StreamChunk(
        delta=delta,
        index=index,
        finish_reason=finish_reason,
        usage=usage,
        metadata=metadata,
    )
