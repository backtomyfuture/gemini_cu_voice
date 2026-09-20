"""
Tool Policy & Structured Contract Module for Gemini Live Computer Use
提供细粒度工具执行策略（只读/写入/高危审查）与结构化工具执行契约
"""
import dataclasses
import enum
import json
import logging
import re
from typing import Any, Dict, Optional, Tuple


class PolicyLevel(str, enum.Enum):
    READ = "read"            # 只读探查，完全无破坏性，无条件自动放行
    WRITE = "write"          # 界面输入或前台切换，有副作用但常规操作，审计记录并放行
    DANGEROUS = "dangerous"  # 存在破坏性或高风险操作，需严密审查/二次确认或在保护模式下拦截


@dataclasses.dataclass
class ToolResultContract:
    """标准结构化工具执行结果契约"""
    ok: bool
    action: str
    status: str             # "success", "denied", "cancelled", "error"
    summary: str
    data: Any = None
    error: Optional[str] = None
    side_effects: str = "none"  # "none", "ui_updated", "app_launched", "navigation", "danger"

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "ok": self.ok,
            "action": self.action,
            "status": self.status,
            "summary": self.summary,
        }
        if self.error:
            d["error"] = self.error
        if self.data is not None:
            d["data"] = self.data
        if self.side_effects != "none":
            d["side_effects"] = self.side_effects
        return d

    def to_gemini_response(self) -> str:
        """格式化为适合向大模型返回的紧凑自然字符串，保证模型精准理解执行情况"""
        status_tag = "【成功】" if self.ok else "【未执行/失败】"
        res = f"{status_tag} {self.action}: {self.summary}"
        if self.error:
            res += f" (原因: {self.error})"
        if self.data:
            data_str = str(self.data)
            if len(data_str) > 3000:
                data_str = data_str[:3000] + "...(内容截断)"
            res += f"\n{data_str}"
        return res


class CancellationToken:
    """支持多协程/多步骤协同取消的令牌"""
    def __init__(self, token_id: str = ""):
        self.token_id = token_id
        self._cancelled = False
        self._cancel_reason = ""

    def cancel(self, reason: str = "User interrupted"):
        self._cancelled = True
        self._cancel_reason = reason

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    @property
    def reason(self) -> str:
        return self._cancel_reason


# 默认只读工具集合
READ_ONLY_TOOLS = {
    "list_apps",
    "get_app_state",
    "browser_get_content",
    "browser_open",
    "browser_search",
    "browser_scroll",
    "browser_list_actions",
}

# 默认写操作工具集合
WRITE_TOOLS = {
    "open_app",
    "click",
    "type_text",
    "browser_click",
    "press_key",
    "scroll",
}

# 高危按键组合 (例如批量删除、关闭系统、强制退出等)
DANGEROUS_KEYS = {
    "cmd+alt+esc",
    "cmd+shift+q",
    "cmd+delete",
    "cmd+backspace",
    "rm -rf",
}

# 疑似 Prompt Injection 注入特征模式
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"system\s*:\s*override", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?guidelines", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+in\s+developer\s+mode", re.IGNORECASE),
]


class ToolPolicyManager:
    """工具安全策略管理器"""
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode

    def classify_tool(self, func_name: str, func_args: Dict[str, Any]) -> Tuple[PolicyLevel, str]:
        """评估并返回工具调用的安全等级与风险说明"""
        # 1. 只读类工具
        if func_name in READ_ONLY_TOOLS:
            return PolicyLevel.READ, "只读探查操作，无破坏性"

        # 2. 审查 press_key 参数
        if func_name == "press_key":
            keys = str(func_args.get("keys", "")).lower().strip()
            for dk in DANGEROUS_KEYS:
                if dk in keys:
                    return PolicyLevel.DANGEROUS, f"按键包含高危组合 '{keys}'，可能导致数据永久丢失或系统退出"
            return PolicyLevel.WRITE, f"常规按键操作 '{keys}'"

        # 3. 审查 type_text 文本（防恶意命令注入）
        if func_name == "type_text":
            text = str(func_args.get("text", ""))
            # 检查 prompt injection 关键词
            for pattern in PROMPT_INJECTION_PATTERNS:
                if pattern.search(text):
                    return PolicyLevel.DANGEROUS, f"输入文本包含可疑的 Prompt 劫持特征: '{text[:40]}...'"
            # 检查危险终端命令特征
            if any(k in text for k in ["rm -rf", "sudo ", "mkfs", "dd if="]):
                return PolicyLevel.DANGEROUS, f"输入文本包含危险破坏性指令: '{text[:40]}...'"
            return PolicyLevel.WRITE, f"文本输入 ({len(text)} 字符)"

        # 4. 审查应用启动
        if func_name == "open_app":
            name = str(func_args.get("name", "") or func_args.get("app", ""))
            if any(k in name.lower() for k in ["terminal", "iterm", "终端", "powershell"]):
                # 终端类应用打开需审计提示
                return PolicyLevel.WRITE, f"启动高权限终端应用: {name}"
            return PolicyLevel.WRITE, f"启动并激活应用程序: {name}"

        # 5. 常规写操作
        if func_name in WRITE_TOOLS:
            return PolicyLevel.WRITE, f"常规界面操作: {func_name}"

        # 6. 未知工具默认作为 WRITE 级别审查
        return PolicyLevel.WRITE, f"自定义/外部扩展工具: {func_name}"

    def check_execution(
        self,
        func_name: str,
        func_args: Dict[str, Any],
        cancellation_token: Optional[CancellationToken] = None
    ) -> Tuple[bool, ToolResultContract]:
        """
        在执行前综合检查策略与取消状态
        返回: (是否允许执行, 结果契约)
        """
        # 1. 检查取消状态
        if cancellation_token and cancellation_token.is_cancelled:
            return False, ToolResultContract(
                ok=False,
                action=func_name,
                status="cancelled",
                summary="用户已打断操作，跳过后续执行",
                error=f"Cancelled: {cancellation_token.reason}"
            )

        # 2. 检查等级与拦截
        level, reason = self.classify_tool(func_name, func_args)
        if level == PolicyLevel.DANGEROUS and self.strict_mode:
            return False, ToolResultContract(
                ok=False,
                action=func_name,
                status="denied",
                summary="高危操作已被安全执行策略拦截",
                error=f"Security Policy Denied: {reason}",
                side_effects="danger"
            )

        # 3. 放行
        return True, ToolResultContract(
            ok=True,
            action=func_name,
            status="allowed",
            summary=f"策略放行: {reason}",
            side_effects="ui_updated" if level == PolicyLevel.WRITE else "none"
        )
