from __future__ import annotations

import asyncio
import os
import re
import shlex
import shutil
import sys
import time
from pathlib import Path
from typing import Any

from closeclaw.agent.tools.base import Tool
from closeclaw.constants import (
    COMMAND_DANGEROUS_BLOCKED,
    COMMAND_DANGEROUS_PATTERN,
    COMMAND_EMPTY,
    COMMAND_INVALID_SYNTAX,
    COMMAND_NOT_ALLOWED,
    COMMAND_SUBSHELL_NOT_ALLOWED,
    COMMAND_TIMEOUT,
    GREP_SUGGESTION,
    SHELL_COMMAND_ALLOWLIST,
    SHELL_DANGEROUS_COMMANDS,
    SHELL_DANGEROUS_PATTERNS,
    SHELL_READ_ONLY_COMMANDS,
    SHELL_REDIRECT_OPERATORS,
    SHELL_RETURN_CODE_ERROR,
    SHELL_SAFE_GIT_SUBCOMMANDS,
    SHELL_SUBSHELL_PATTERNS,
    SHELL_SYSTEM_DIRECTORIES,
    TOOL_SHELL_ERROR,
    TOOL_SHELL_EXEC,
    TOOL_SHELL_RETURN,
    TOOL_SHELL_STDERR,
    TOOL_SHELL_STDOUT,
    SHELL_COMMANDER_INIT,
)
from closeclaw.logger import app_logger as logger

# 编译危险命令模式
PIPELINE_PATTERNS_COMPILED = tuple(
    (re.compile(pattern, re.IGNORECASE), reason)
    for pattern, reason in SHELL_DANGEROUS_PATTERNS
)

# Shell 命令常量
SHELL_CMD_RM = "rm"
SHELL_CMD_GREP = "grep"


class ShellCommandResult:
    """Shell命令执行结果

    封装Shell命令的执行结果，包括返回码、标准输出和标准错误。

    Attributes:
        return_code: 命令返回码，0表示成功
        stdout: 标准输出内容
        stderr: 标准错误内容
    """

    def __init__(self, return_code: int, stdout: str, stderr: str) -> None:
        self.return_code = return_code
        self.stdout = stdout
        self.stderr = stderr

    def to_dict(self) -> dict[str, Any]:
        """转换为字典格式"""
        return {
            "return_code": self.return_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
        }

    def __repr__(self) -> str:
        return f"ShellCommandResult(return_code={self.return_code}, stdout={self.stdout!r}, stderr={self.stderr!r})"


class CommandGroup:
    """命令组

    表示一个命令组，包含多个通过管道连接的命令。

    Attributes:
        commands: 命令列表
        operator: 逻辑操作符（如 &&, ||, ;）
    """

    __slots__ = ("commands", "operator")

    def __init__(self, commands: list[str], operator: str | None = None) -> None:
        self.commands = commands
        self.operator = operator


def _is_outside_single_quotes(command: str, pos: int) -> bool:
    """检查位置是否在单引号外部

    Args:
        command: 命令字符串
        pos: 要检查的位置

    Returns:
        如果在单引号外部返回True，否则返回False
    """
    in_single = False
    i = 0
    while i < pos:
        char = command[i]
        if char == "\\" and not in_single and i + 1 < len(command):
            i += 2
            continue
        if char == "'":
            in_single = not in_single
        i += 1
    return not in_single


def _has_subshell(command: str) -> str | None:
    """检查命令是否包含子shell

    Args:
        command: 命令字符串

    Returns:
        如果包含子shell返回模式字符串，否则返回None
    """
    for pattern in SHELL_SUBSHELL_PATTERNS:
        start = 0
        while True:
            pos = command.find(pattern, start)
            if pos == -1:
                break
            if _is_outside_single_quotes(command, pos):
                return pattern
            start = pos + 1
    return None


def _split_by_logical_operators(command: str) -> list[tuple[str, str | None]]:
    """按逻辑操作符分割命令

    使用 shlex 安全解析命令，识别 &&, ||, ; 等逻辑操作符。

    Args:
        command: 命令字符串

    Returns:
        [(命令段, 前置操作符), ...]
    """
    parts: list[tuple[str, str | None]] = []
    current_parts: list[str] = []
    current_op: str | None = None

    # 使用 shlex 分割，传入 punctuation_chars 到构造函数
    lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True

    for token in lexer:
        if token in ("&&", "||", ";"):
            if current_parts:
                parts.append((" ".join(current_parts), current_op))
                current_parts = []
            current_op = token
        else:
            current_parts.append(token)

    if current_parts:
        parts.append((" ".join(current_parts), current_op))

    return parts


def _parse_command(command: str) -> list[CommandGroup]:
    """解析命令字符串为命令组

    使用 shlex 安全解析命令字符串，支持管道和逻辑操作符。

    Args:
        command: 命令字符串

    Returns:
        命令组列表
    """
    logical_parts = _split_by_logical_operators(command)
    groups: list[CommandGroup] = []

    for segment, op in logical_parts:
        # 按管道分割
        pipeline_cmds = [p.strip() for p in segment.split("|") if p.strip()]
        if pipeline_cmds:
            groups.append(CommandGroup(pipeline_cmds, op))

    return groups


def _is_blocked_command(cmd: str) -> bool:
    """检查命令是否被阻止

    Args:
        cmd: 命令名称

    Returns:
        如果被阻止返回True
    """
    return cmd in SHELL_DANGEROUS_COMMANDS


def _is_dangerous_rm(cmd_parts: list[str]) -> bool:
    """检查rm命令是否危险

    检查rm命令是否包含 -rf 等危险标志。

    Args:
        cmd_parts: 命令参数列表

    Returns:
        如果危险返回True
    """
    if not cmd_parts or cmd_parts[0] != SHELL_CMD_RM:
        return False
    flags = "".join(part for part in cmd_parts[1:] if part.startswith("-"))
    return "r" in flags and "f" in flags


def _is_safe_path(path_str: str, project_root: Path) -> tuple[bool, str | None]:
    """检查路径是否在项目目录内

    使用 pathlib 的 is_relative_to 进行安全检查。

    Args:
        path_str: 路径字符串
        project_root: 项目根目录

    Returns:
        (是否安全, 不安全原因或None)
    """
    if path_str in ("*", ".", ".."):
        return False, f"dangerous path pattern: {path_str}"

    try:
        if path_str.startswith("/"):
            resolved = Path(path_str).resolve()
        else:
            resolved = (project_root / path_str).resolve()
    except (OSError, ValueError):
        return False, f"invalid path: {path_str}"

    # 检查是否是根目录
    if resolved == resolved.parent:
        return False, "targeting root directory"

    # 检查是否在项目目录内
    try:
        resolved.relative_to(project_root.resolve())
        return True, None
    except ValueError:
        # 在项目目录外，检查是否是系统目录
        parts = resolved.parts
        if len(parts) >= 2 and parts[1] in SHELL_SYSTEM_DIRECTORIES:
            return False, f"targeting system directory: {resolved}"
        return False, f"targeting path outside project: {resolved}"


def _is_dangerous_rm_path(cmd_parts: list[str], project_root: Path) -> tuple[bool, str]:
    """检查rm命令的路径是否危险

    Args:
        cmd_parts: 命令参数列表
        project_root: 项目根目录

    Returns:
        (是否危险, 原因)
    """
    if not cmd_parts or cmd_parts[0] != SHELL_CMD_RM:
        return False, ""

    path_args = [p for p in cmd_parts[1:] if not p.startswith("-")]
    for path_arg in path_args:
        is_safe, reason = _is_safe_path(path_arg, project_root)
        if not is_safe:
            return True, f"rm {reason}"

    return False, ""


def _check_pipeline_patterns(full_command: str) -> str | None:
    """检查命令是否匹配危险模式

    Args:
        full_command: 完整命令字符串

    Returns:
        如果匹配返回原因，否则返回None
    """
    for pattern, reason in PIPELINE_PATTERNS_COMPILED:
        if pattern.search(full_command):
            return reason
    return None


def _is_dangerous_command(cmd_parts: list[str]) -> tuple[bool, str]:
    """检查命令是否危险

    Args:
        cmd_parts: 命令参数列表

    Returns:
        (是否危险, 原因)
    """
    if not cmd_parts:
        return False, ""

    base_cmd = cmd_parts[0]

    if _is_blocked_command(base_cmd):
        return True, f"blocked command: {base_cmd}"

    if _is_dangerous_rm(cmd_parts):
        return True, "rm with dangerous flags"

    return False, ""


def _validate_segment(
    segment: str, available_commands: str, check_patterns: bool = True
) -> str | None:
    """验证命令段

    检查命令是否在允许列表中，是否包含危险操作。

    Args:
        segment: 命令段
        available_commands: 可用命令列表字符串
        check_patterns: 是否检查危险模式（execute 已全局检查时可设为 False）

    Returns:
        如果验证失败返回错误消息，否则返回None
    """
    try:
        cmd_parts = shlex.split(segment)
    except ValueError:
        return COMMAND_INVALID_SYNTAX.format(segment=segment)

    if not cmd_parts:
        return None

    base_cmd = cmd_parts[0]

    if base_cmd not in SHELL_COMMAND_ALLOWLIST:
        suggestion = GREP_SUGGESTION if base_cmd == SHELL_CMD_GREP else ""
        return COMMAND_NOT_ALLOWED.format(
            cmd=base_cmd, suggestion=suggestion, available=available_commands
        )

    is_dangerous, reason = _is_dangerous_command(cmd_parts)
    if is_dangerous:
        return COMMAND_DANGEROUS_BLOCKED.format(cmd=base_cmd, reason=reason)

    if check_patterns:
        if pattern_reason := _check_pipeline_patterns(segment):
            return COMMAND_DANGEROUS_PATTERN.format(reason=pattern_reason)

    return None


def _has_redirect_operators(parts: list[str]) -> bool:
    """检查是否包含重定向操作符

    Args:
        parts: 命令参数列表

    Returns:
        如果包含重定向操作符返回True
    """
    return any(p in SHELL_REDIRECT_OPERATORS for p in parts)


def _requires_approval(command: str) -> bool:
    """检查命令是否需要审批

    只读命令不需要审批，其他命令需要审批。

    Args:
        command: 命令字符串

    Returns:
        如果需要审批返回True
    """
    if not command.strip():
        return True

    try:
        groups = _parse_command(command)
    except (ValueError, IndexError):
        return True

    has_commands = False
    for group in groups:
        for segment in group.commands:
            segment = segment.strip()
            if not segment:
                continue
            try:
                parts = shlex.split(segment)
            except ValueError:
                return True

            if not parts:
                continue

            if _has_redirect_operators(parts):
                return True

            has_commands = True
            base_cmd = parts[0]
            if base_cmd in SHELL_READ_ONLY_COMMANDS:
                continue

            if base_cmd == "git" and len(parts) > 1:
                if parts[1] in SHELL_SAFE_GIT_SUBCOMMANDS:
                    continue

            return True

    return not has_commands


class ShellCommander:
    """Shell命令执行器

    负责执行Shell命令，包含安全检查、超时控制和结果处理。

    Attributes:
        project_root: 项目根目录路径
        timeout: 命令执行超时时间（秒），这是整个命令管道执行的总超时时间
    """

    __slots__ = ("project_root", "timeout")

    def __init__(self, project_root: str = ".", timeout: int = 30) -> None:
        self.project_root = Path(project_root).resolve()
        self.timeout = timeout
        logger.info(SHELL_COMMANDER_INIT.format(root=self.project_root))

    async def _execute_pipeline(self, segments: list[str]) -> tuple[int, bytes, bytes]:
        """执行管道命令

        执行通过管道连接的多个命令。超时时间是整个管道的总超时时间，
        不是单个命令的超时。如果管道中前面的命令耗时较长，后面的命令
        可能可用的剩余时间较少。

        Args:
            segments: 命令段列表

        Returns:
            (返回码, 标准输出, 标准错误)

        Raises:
            TimeoutError: 命令执行超时（总执行时间超过 timeout）
        """
        start_time = time.monotonic()
        input_data: bytes | None = None
        all_stderr: list[bytes] = []
        last_return_code = 0

        env = dict(os.environ)
        if sys.platform == "win32":
            git_bin = r"C:\Program Files\Git\usr\bin"
            if os.path.isdir(git_bin) and git_bin not in env["PATH"]:
                env["PATH"] = f"{git_bin};{env['PATH']}"

        for segment in segments:
            elapsed = time.monotonic() - start_time
            remaining_timeout = self.timeout - elapsed
            if remaining_timeout <= 0:
                raise TimeoutError

            cmd_parts = shlex.split(segment)
            executable = shutil.which(cmd_parts[0], path=env["PATH"])
            if not executable:
                executable = cmd_parts[0]

            proc = await asyncio.create_subprocess_exec(
                executable,
                *cmd_parts[1:],
                stdin=asyncio.subprocess.PIPE if input_data is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.project_root,
                env=env,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=input_data), timeout=remaining_timeout
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise

            last_return_code = (
                proc.returncode
                if proc.returncode is not None
                else SHELL_RETURN_CODE_ERROR
            )
            input_data = stdout

            if stderr:
                all_stderr.append(stderr)

        return last_return_code, input_data or b"", b"".join(all_stderr)

    async def execute(self, command: str) -> ShellCommandResult:
        """执行Shell命令

        执行Shell命令，包含完整的安全检查和错误处理。

        Args:
            command: 要执行的命令

        Returns:
            ShellCommandResult: 命令执行结果
        """
        logger.info(TOOL_SHELL_EXEC.format(cmd=command))
        try:
            if subshell_pattern := _has_subshell(command):
                err_msg = COMMAND_SUBSHELL_NOT_ALLOWED.format(pattern=subshell_pattern)
                logger.error(err_msg)
                return ShellCommandResult(
                    return_code=SHELL_RETURN_CODE_ERROR, stdout="", stderr=err_msg
                )

            if pattern_reason := _check_pipeline_patterns(command):
                err_msg = COMMAND_DANGEROUS_PATTERN.format(reason=pattern_reason)
                logger.error(err_msg)
                return ShellCommandResult(
                    return_code=SHELL_RETURN_CODE_ERROR,
                    stdout="",
                    stderr=err_msg,
                )

            groups = _parse_command(command)
            if not groups:
                return ShellCommandResult(
                    return_code=SHELL_RETURN_CODE_ERROR,
                    stdout="",
                    stderr=COMMAND_EMPTY,
                )

            available_commands = ", ".join(sorted(SHELL_COMMAND_ALLOWLIST))
            for group in groups:
                for segment in group.commands:
                    # 跳过已在全局检查的模式
                    validation_error = _validate_segment(
                        segment, available_commands, check_patterns=False
                    )
                    if validation_error:
                        logger.error(validation_error)
                        return ShellCommandResult(
                            return_code=SHELL_RETURN_CODE_ERROR,
                            stdout="",
                            stderr=validation_error,
                        )
                    try:
                        cmd_parts = shlex.split(segment)
                    except ValueError:
                        continue
                    is_dangerous, reason = _is_dangerous_rm_path(
                        cmd_parts, self.project_root
                    )
                    if is_dangerous:
                        err_msg = COMMAND_DANGEROUS_BLOCKED.format(
                            cmd=cmd_parts[0], reason=reason
                        )
                        logger.error(err_msg)
                        return ShellCommandResult(
                            return_code=SHELL_RETURN_CODE_ERROR,
                            stdout="",
                            stderr=err_msg,
                        )

            all_stdout: list[str] = []
            all_stderr: list[str] = []
            last_return_code = 0

            for group in groups:
                should_run = True
                if group.operator == "&&":
                    should_run = last_return_code == 0
                elif group.operator == "||":
                    should_run = last_return_code != 0

                if not should_run:
                    continue

                return_code, stdout, stderr = await self._execute_pipeline(
                    group.commands
                )
                last_return_code = return_code

                stdout_str = stdout.decode("utf-8", errors="replace").strip()
                stderr_str = stderr.decode("utf-8", errors="replace").strip()

                if stdout_str:
                    all_stdout.append(stdout_str)
                if stderr_str:
                    all_stderr.append(stderr_str)

            final_stdout = "\n".join(all_stdout)
            final_stderr = "\n".join(all_stderr)

            logger.info(TOOL_SHELL_RETURN.format(code=last_return_code))
            if final_stdout:
                logger.info(TOOL_SHELL_STDOUT.format(stdout=final_stdout))
            if final_stderr:
                logger.warning(TOOL_SHELL_STDERR.format(stderr=final_stderr))

            return ShellCommandResult(
                return_code=last_return_code,
                stdout=final_stdout,
                stderr=final_stderr,
            )
        except TimeoutError:
            msg = COMMAND_TIMEOUT.format(cmd=command, timeout=self.timeout)
            logger.error(msg)
            return ShellCommandResult(
                return_code=SHELL_RETURN_CODE_ERROR, stdout="", stderr=msg
            )
        except Exception as e:
            logger.error(TOOL_SHELL_ERROR.format(error=e))
            return ShellCommandResult(
                return_code=SHELL_RETURN_CODE_ERROR, stdout="", stderr=str(e)
            )


class ShellTool(Tool):
    """命令行工具

    提供安全的Shell命令执行功能，支持命令白名单、危险命令检测、
    超时控制和管道命令执行。

    Attributes:
        timeout: 命令执行总超时时间（秒），覆盖整个命令管道执行过程
    """

    timeout: int | None = 30

    def __init__(self, project_root: str = ".", timeout: int = 30) -> None:
        self._commander = ShellCommander(project_root, timeout)
        self.timeout = timeout

    @property
    def name(self) -> str:
        """工具名称"""
        return "execute_shell"

    @property
    def description(self) -> str:
        """工具描述"""
        return (
            "执行Shell命令。支持管道和逻辑操作符（&&, ||）。"
            "只允许白名单中的命令，会自动阻止危险操作。"
            "可用命令包括：ls, cat, echo, pwd, git, find, rg, grep等。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """工具参数定义

        Returns:
            符合JSON Schema格式的参数定义
        """
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的Shell命令",
                }
            },
            "required": ["command"],
        }

    async def exec(self, **params) -> str:
        """执行Shell命令

        Args:
            **params: 包含command参数的字典

        Returns:
            JSON格式的命令执行结果
        """
        import json

        command = params.get("command", "")
        result = await self._commander.execute(command)
        return json.dumps(result.to_dict(), ensure_ascii=False)

    def requires_approval(self, command: str) -> bool:
        """检查命令是否需要用户审批

        Args:
            command: 命令字符串

        Returns:
            如果需要审批返回True
        """
        return _requires_approval(command)
