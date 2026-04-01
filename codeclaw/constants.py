from enum import IntEnum

# 供应商信息
OPENAI_DEFAULT_ENDPOINT = "https://api.openai.com/v1"

# 工具信息
## 命令行工具
# region


class ExitCode(IntEnum):
    """Shell 命令执行返回码"""

    SUCCESS = 0
    VALIDATION_ERROR = 125
    EXECUTION_ERROR = 126
    TIMEOUT = 124
    GENERAL_ERROR = 1


# Shell 执行相关常量
SHELL_RETURN_CODE_ERROR = ExitCode.GENERAL_ERROR
SHELL_CMD_RM = "rm"
SHELL_CMD_GREP = "grep"
SHELL_REDIRECT_OPERATORS = frozenset([">", ">>", "<", "<<", ">", ">>", "&>", "&>>"])
SHELL_SUBSHELL_PATTERNS = frozenset(["$(", "`", "<(", ">("])
SHELL_SYSTEM_DIRECTORIES = frozenset(
    [
        "bin",
        "boot",
        "dev",
        "etc",
        "lib",
        "lib64",
        "proc",
        "root",
        "sbin",
        "sys",
        "usr",
        "var",
    ]
)
SHELL_READ_ONLY_COMMANDS = frozenset(
    [
        "ls",
        "cat",
        "echo",
        "pwd",
        "head",
        "tail",
        "wc",
        "sort",
        "uniq",
        "cut",
        "tr",
        "find",
        "rg",
        "grep",
    ]
)
SHELL_SAFE_GIT_SUBCOMMANDS = frozenset(
    ["status", "log", "diff", "show", "branch", "remote", "config", "ls-files"]
)

# 构建系统目录正则模式
_SYSTEM_DIRS_PATTERN = "|".join(SHELL_SYSTEM_DIRECTORIES)

SHELL_COMMAND_ALLOWLIST = frozenset(
    {
        "ls",
        "rg",
        "cat",
        "git",
        "echo",
        "pwd",
        "pytest",
        "mypy",
        "ruff",
        "uv",
        "find",
        "pre-commit",
        "rm",
        "cp",
        "mv",
        "mkdir",
        "rmdir",
        "wc",
        "head",
        "tail",
        "sort",
        "uniq",
        "cut",
        "tr",
        "xargs",
        "awk",
        "sed",
        "tee",
    }
)

SHELL_DANGEROUS_COMMANDS = frozenset(
    {
        "dd",
        "mkfs",
        "mkfs.ext4",
        "mkfs.ext3",
        "mkfs.xfs",
        "mkfs.btrfs",
        "mkfs.vfat",
        "fdisk",
        "parted",
        "shred",
        "wipefs",
        "mkswap",
        "swapon",
        "swapoff",
        "mount",
        "umount",
        "insmod",
        "rmmod",
        "modprobe",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "init",
        "telinit",
        "systemctl",
        "service",
        "chroot",
        "nohup",
        "disown",
        "crontab",
        "at",
        "batch",
    }
)

SHELL_DANGEROUS_PATTERNS = (
    (r"rm\s+.*-[rf]+\s+/($|\s)", "rm with root path"),
    (rf"rm\s+.*-[rf]+\s+/({_SYSTEM_DIRS_PATTERN})($|/|\s)", "rm with system directory"),
    (r"rm\s+.*-[rf]+\s+~($|\s)", "rm with home directory"),
    (r"rm\s+.*-[rf]+\s+\*", "rm with wildcard"),
    (r"rm\s+.*-[rf]+\s+\.\.", "rm with parent directory"),
    (r"dd\s+.*of=/dev/", "dd writing to device"),
    (r">\s*/dev/sd[a-z]", "redirect to disk device"),
    (r">\s*/dev/nvme", "redirect to nvme device"),
    (r">\s*/dev/null.*<", "null device manipulation"),
    (r"chmod\s+.*-R\s+777\s+\/", "recursive 777 on root"),
    (r"chmod\s+.*777\s+/($|\s)", "777 on root"),
    (r"chown\s+.*-R\s+.*\s+/($|\s)", "recursive chown on root"),
    (r":\(\)\s*\{.*:\s*\|", "fork bomb pattern"),
    (r"mv\s+.*\s+/dev/null", "move to /dev/null"),
    (r"ln\s+-[sf]+\s+/dev/null", "symlink to /dev/null"),
    (r"cat\s+.*/dev/zero", "cat /dev/zero"),
    (r"cat\s+.*/dev/random", "cat /dev/random"),
    (r">\s*/etc/passwd", "overwrite passwd"),
    (r">\s*/etc/shadow", "overwrite shadow"),
    (r">\s*/etc/sudoers", "overwrite sudoers"),
    (r"echo\s+.*>\s*/etc/", "write to /etc"),
    (
        r"python.*-c.*(import\s+os|__import__\s*\(\s*['\"]os['\"]\s*\))",
        "python os import in command",
    ),
    (r"perl\s+-e", "perl one-liner"),
    (r"ruby\s+-e", "ruby one-liner"),
    (r"nc\s+-[el]", "netcat listener"),
    (r"ncat\s+-[el]", "ncat listener"),
    (r"/dev/tcp/", "bash tcp device"),
    (r"eval\s+", "eval command"),
    (r"exec\s+[0-9]+<>", "exec file descriptor manipulation"),
    (r"awk\s+.*system\s*\(", "awk system() call"),
    (r"awk\s+.*getline\s*[<|]", "awk getline file/pipe execution"),
    (r"sed\s+.*s(.).*?\1.*?\1[gip]*e[gip]*", "sed execute flag"),
    (r"xargs\s+.*(rm|chmod|chown|mv|dd|mkfs)", "xargs with destructive command"),
    (r"xargs\s+-I.*sh", "xargs shell execution"),
    (r"xargs\s+.*bash", "xargs bash execution"),
)

# Shell 错误消息模板
COMMAND_INVALID_SYNTAX = "命令语法无效: {segment}"
COMMAND_NOT_ALLOWED = "命令 '{cmd}' 不在允许列表中。{suggestion}可用命令: {available}"
COMMAND_DANGEROUS_BLOCKED = "命令 '{cmd}' 被阻止: {reason}"
COMMAND_EMPTY = "命令为空"
COMMAND_SUBSHELL_NOT_ALLOWED = "子shell模式 '{pattern}' 不允许"
COMMAND_DANGEROUS_PATTERN = "命令包含危险模式: {reason}"
COMMAND_TIMEOUT = "命令执行超时: {cmd} (超时时间: {timeout}秒)"
GREP_SUGGESTION = "请使用 'rg' 替代 'grep'. "

# Shell 日志消息模板
SHELL_COMMANDER_INIT = "ShellCommander 初始化完成，项目根目录: {root}"
TOOL_SHELL_EXEC = "执行Shell命令: {cmd}"
TOOL_SHELL_RETURN = "Shell命令返回码: {code}"
TOOL_SHELL_STDOUT = "Shell命令输出: {stdout}"
TOOL_SHELL_STDERR = "Shell命令错误: {stderr}"
TOOL_SHELL_ERROR = "Shell命令执行错误: {error}"
# endregion
