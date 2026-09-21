# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "google-genai>=2.20.0",
#     "mcp>=1.0.0",
#     "sounddevice>=0.5.0",
#     "numpy>=1.24.0",
#     "python-dotenv>=1.0.0",
# ]
# ///
import argparse
import asyncio
import collections
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional, Any, Dict, List, Set, Union, Callable

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import numpy as np
import sounddevice as sd

from tool_policy import (
    ToolPolicyManager,
    ToolResultContract,
    CancellationToken,
    PolicyLevel,
    is_browser_error
)
from ego_browser_client import (
    browser_open,
    browser_search,
    browser_get_content,
    browser_list_actions,
    browser_click,
    browser_scroll,
    browser_close,
    get_browser_function_declarations
)

# 加载配置
CUR_DIR = Path(__file__).parent
env_path = CUR_DIR / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

# 日志持久化文件
LOG_FILE = CUR_DIR / "gemini_live_cu.log"
NO_LOG_CONTENT = False


class ConversationMemory:
    """本地对话上下文记忆池，支持多轮持续累积与断线重连自动回灌"""
    def __init__(self, max_turns: int = 15):
        self.max_turns = max_turns
        self.history = collections.deque(maxlen=max_turns * 2)

    def record_turn(self, user_text: str = "", model_text: str = "", tool_summary: str = ""):
        parts_user = []
        if user_text and user_text.strip():
            parts_user.append(types.Part.from_text(text=user_text.strip()))
        if parts_user:
            self.history.append(types.Content(role="user", parts=parts_user))

        parts_model = []
        if tool_summary and tool_summary.strip():
            parts_model.append(types.Part.from_text(text=f"[已执行操作记录]: {tool_summary.strip()}"))
        if model_text and model_text.strip():
            parts_model.append(types.Part.from_text(text=model_text.strip()))
        if parts_model:
            self.history.append(types.Content(role="model", parts=parts_model))

    def get_prefill_turns(self) -> list:
        return list(self.history)


class ResumptionHandleExpiredError(ConnectionError):
    """当使用上次保存的会话恢复句柄建连握手失败时抛出，指示 handle 已失效需降级为记忆注入"""
    pass


class GoAwayReconnectError(ConnectionError):
    """服务端下发 GoAway 信令触发的平滑快速重连异常"""
    def __init__(self, time_left: Optional[str] = None):
        super().__init__(f"Server issued GoAway signal (time_left={time_left})")
        self.time_left = time_left


STATE_LISTENING = "LISTENING"  # 空闲听用户说话
STATE_THINKING = "THINKING"    # 用户已说完，正在等待 Gemini 思考/下发动作
STATE_EXECUTING = "EXECUTING"  # kimi-cu 工具正在物理执行中
STATE_SPEAKING = "SPEAKING"    # 扬声器正在平滑播报中


class TurnController:
    """集中式轮次生命周期控制器，管理单调递增 turn_id、当前取消令牌、后台工具任务、播放结束隔离与会话状态"""
    def __init__(self, use_interaction_status: bool = False, policy_manager: Optional[Any] = None):
        self.current_turn_id: int = 0
        self.cancellation_token: CancellationToken = CancellationToken("turn_0")
        self.active_tool_task: Optional[asyncio.Task] = None
        self.active_tool_tasks: List[asyncio.Task] = []
        self.waiting_tool_summary: bool = False
        self.has_active_tool: bool = False
        self.is_interrupted: bool = False
        self.state: str = STATE_LISTENING
        self.state_start_time: float = time.time()
        self.use_interaction_status = use_interaction_status
        self.policy_manager = policy_manager

    def set_state(self, new_state: str):
        """唯一的会话状态迁移入口；状态与进入时刻都集中在此"""
        if self.state != new_state:
            self.state = new_state
            self.state_start_time = time.time()
            log_event("STATE", f"Transitioned to {new_state}")

    def new_turn(self, reason: str = "new_turn") -> int:
        """开启新轮次：递增 turn_id，生成全新有效 CancellationToken，取消可能残余的工具任务，按指令轮次重置策略预算"""
        self.current_turn_id += 1
        self._cancel_tool_tasks()
        self.cancellation_token = CancellationToken(f"turn_{self.current_turn_id}_{time.time():.3f}")
        self.has_active_tool = False
        self.waiting_tool_summary = False
        self.is_interrupted = False
        if self.policy_manager:
            self.policy_manager.reset_turn()
        return self.current_turn_id

    def _cancel_tool_tasks(self):
        tasks = list(self.active_tool_tasks)
        if self.active_tool_task and self.active_tool_task not in tasks:
            tasks.append(self.active_tool_task)
        for task in tasks:
            if task and not task.done():
                task.cancel()
        self.active_tool_tasks.clear()
        self.active_tool_task = None

    def register_tool_task(self, task: asyncio.Task):
        """登记同轮多个 in-flight 工具任务，后一次调用不覆盖前一次。"""
        self.has_active_tool = True
        self.active_tool_task = task
        if task not in self.active_tool_tasks:
            self.active_tool_tasks.append(task)

    def interrupt(self, reason: str = "interrupted"):
        """处理打断：取消当前 Token 与工具任务，重置状态标记并回到 LISTENING"""
        self.is_interrupted = True
        self.cancellation_token.cancel(reason)
        self._cancel_tool_tasks()
        self.has_active_tool = False
        self.waiting_tool_summary = False
        self.set_state(STATE_LISTENING)

    def finish_active_tool(self, turn_id: int, task: Optional[asyncio.Task] = None):
        """关键修复：仅当轮次与任务对象精确匹配时才清理 has_active_tool，防止旧任务退出冲刷新任务状态"""
        if self.current_turn_id != turn_id:
            return
        if task is not None and task in self.active_tool_tasks:
            self.active_tool_tasks.remove(task)
        if task is None or self.active_tool_task is task:
            if self.active_tool_task is task:
                self.active_tool_task = None
        if self.active_tool_tasks:
            self.has_active_tool = True
            if self.active_tool_task is None:
                self.active_tool_task = self.active_tool_tasks[-1]
        else:
            self.has_active_tool = False
            if task is None or self.active_tool_task is task or self.active_tool_task is None:
                self.active_tool_task = None

    def is_current_turn(self, turn_id: int) -> bool:
        """检查指定 turn_id 是否仍为当前最新轮次（用于丢弃迟到的旧轮次回调）"""
        return self.current_turn_id == turn_id

    def is_interaction_idle(self, response: Any) -> bool:
        """默认模型以 turn_complete 为空闲；Extended Thinking 仅在 interaction_status=IDLE 时为空闲。"""
        server_content = getattr(response, "server_content", None)
        if self.use_interaction_status:
            status = getattr(response, "interaction_status", None)
            if not status and server_content is not None:
                status = getattr(server_content, "interaction_status", None)
            return status == "IDLE"
        return bool(server_content and getattr(server_content, "turn_complete", False))


AUDIO_STREAM_END = b"__AUDIO_STREAM_END__"


class AudioTurnQueue:
    """音频 turn 队列：PCM 可丢，当前 turn 的结束信号不可被溢出挤掉。"""

    END = AUDIO_STREAM_END

    def __init__(self, maxsize: int = 250):
        self._maxsize = maxsize
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=maxsize)

    def _drain_nowait(self) -> list:
        items = []
        while True:
            try:
                items.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return items

    def _restore(self, items: list) -> None:
        for item in items:
            self._queue.put_nowait(item)

    @staticmethod
    def _drop_oldest_pcm(items: list) -> bool:
        for i, item in enumerate(items):
            if item != AUDIO_STREAM_END and item is not None:
                items.pop(i)
                return True
        return False

    def enqueue_audio(self, chunk_bytes: bytes) -> None:
        if chunk_bytes == AUDIO_STREAM_END:
            self.finish_turn()
            return
        try:
            self._queue.put_nowait(chunk_bytes)
            return
        except asyncio.QueueFull:
            items = self._drain_nowait()
            self._drop_oldest_pcm(items)
            if len(items) < self._maxsize:
                items.append(chunk_bytes)
            self._restore(items)

    def finish_turn(self) -> None:
        try:
            self._queue.put_nowait(AUDIO_STREAM_END)
            return
        except asyncio.QueueFull:
            items = self._drain_nowait()
            if not self._drop_oldest_pcm(items) and AUDIO_STREAM_END in items:
                self._restore(items)
                return
            if len(items) < self._maxsize:
                items.append(AUDIO_STREAM_END)
            self._restore(items)

    def discard_turn(self) -> None:
        self._drain_nowait()

    async def get(self):
        return await self._queue.get()

    def close(self) -> None:
        items = self._drain_nowait()
        if len(items) >= self._maxsize:
            self._drop_oldest_pcm(items)
        if len(items) < self._maxsize:
            items.append(None)
        self._restore(items)


def update_env_file(path: Path, updates: dict):
    """安全更新 .env 文件中的指定键值对，保留现有注释与其它配置"""
    lines = []
    existing_keys = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and "=" in stripped:
                    key = stripped.split("=", 1)[0].strip()
                    if key in updates:
                        lines.append(f"{key}={updates[key]}\n")
                        existing_keys.add(key)
                        continue
                lines.append(line)
    for k, v in updates.items():
        if k not in existing_keys:
            lines.append(f"{k}={v}\n")
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def log_event(category: str, message: str):
    """写入结构化历史记录文件供排查与对比，支持内容隐私脱敏"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if NO_LOG_CONTENT and category in ["USER_TRANSCRIPT", "MODEL_TRANSCRIPT", "MCP_RESULT", "BROWSER_RESULT", "USER_VOICE"]:
        message = f"[{len(message)} chars - content redacted for privacy]"
    line = f"[{ts}] [{category}] {message}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def clean_ax_text(raw_text: str, max_chars: int = 5000) -> str:
    """智能解析并精简 AX 控件树：保留节点编号 [index]、控件角色与标题/内容，过滤系统菜单栏与纯空容器"""
    header = []
    lines = raw_text.splitlines()

    nodes = []
    node_pattern = re.compile(r'^\s*-\s*\[(\d+)\]\s+(AX\w+)(.*?)$')
    current_node = None
    in_menubar = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("```"):
            continue

        if any(stripped.startswith(prefix) for prefix in ['#', '- app:', '- window_title:', '- bundle_id:']):
            header.append(stripped)
            continue

        m = node_pattern.match(line)
        if m:
            if current_node:
                nodes.append(current_node)
                current_node = None

            idx, role, rest = m.groups()
            if role == 'AXMenuBar':
                in_menubar = True
            elif in_menubar and (role == 'AXMenuBarItem' or 'AXMenu' in role):
                pass
            else:
                in_menubar = False
                current_node = {
                    'idx': idx,
                    'role': role,
                    'lines': [rest]
                }
        else:
            if current_node and not in_menubar:
                current_node['lines'].append(stripped)

    if current_node:
        nodes.append(current_node)

    pure_structural = {'AXSplitGroup', 'AXSplitter', 'AXScrollArea', 'AXScrollBar', 'AXGroup', 'AXUnknown'}

    formatted_items = []
    for n in nodes:
        idx = n['idx']
        role = n['role']
        full_text = ' '.join(n['lines']).strip()

        clean_line = re.sub(r'actions=\[[^\]]*\]', '', full_text)
        clean_line = re.sub(r'@-?\d+,-?\d+\s+\d+×\d+', '', clean_line)

        label_match = re.search(r'\(([^)]+)\)', clean_line)
        label = label_match.group(1).strip() if label_match else ''
        if label in ['disabled', 'enabled']:
            label = ''

        val_match = re.search(r'=\s*\"([^\"]*)\"', clean_line)
        val = val_match.group(1).strip() if val_match else ''
        if not val:
            val_unquoted = re.search(r'=\s*(\S+)', clean_line)
            if val_unquoted and not val_unquoted.group(1).startswith('actions='):
                val = val_unquoted.group(1).strip()

        clean_text_line = clean_line
        if label:
            clean_text_line = clean_text_line.replace(f'({label})', '')
        if val:
            clean_text_line = clean_text_line.replace(f'= "{val}"', '').replace(f'={val}', '')

        help_match = re.search(r'help=\"([^\"]+)\"', clean_text_line)
        help_text = help_match.group(1).strip() if help_match else ''
        clean_text_line = re.sub(r'help=\"[^\"]*\"', '', clean_text_line)

        quotes = re.findall(r'\"([^\"]+)\"', clean_text_line)
        text = ' | '.join(q.strip() for q in quotes if len(q.strip()) > 0)

        desc = text or val or label or help_text
        if role in pure_structural and not desc:
            continue

        item_parts = [f"[{idx}]", role]
        if label:
            item_parts.append(f"({label})")
        if val and val != label:
            item_parts.append(f'= "{val}"')
        if text and text != label and text != val:
            t = text.replace('\n', ' ').strip()
            if len(t) > 130:
                t = t[:130] + "..."
            item_parts.append(f'"{t}"')
        elif not text and not val and help_text and help_text != label:
            item_parts.append(f'help: "{help_text}"')

        formatted_items.append(" ".join(item_parts))

    if not formatted_items:
        return "\n".join(header) + "\n\n【提示】：当前应用暂无打开的前台主窗口或未加载出控件。"

    res = "\n".join(header) + f"\n\n【桌面应用窗口与控件列表（共 {len(formatted_items)} 项）】:\n" + "\n".join(formatted_items)
    if len(res) > max_chars:
        res = res[:max_chars] + "\n...(部分控件已精简)"
    return res


def format_app_list(raw_json: str) -> str:
    """将 list_apps 返回的 JSON 格式化为直观易懂的应用列表"""
    try:
        import json
        data = json.loads(raw_json)
        apps = data.get("apps", [])
        lines = ["【当前正在运行的应用程序】:"]
        ignore_system = {"WindowManager", "Dock", "SystemUIServer", "loginwindow"}
        for a in apps:
            name = a.get("name", "")
            bid = a.get("bundle_id", "")
            if name and name not in ignore_system:
                lines.append(f"- {name} (bundle_id: \"{bid}\", pid: {a.get('pid')})")
        return "\n".join(lines)
    except Exception:
        return raw_json[:800]


def format_tool_result(func_name: str, raw_text: str) -> str:
    """智能分发各工具的输出，杜绝把所有工具误送进 AX 树清洗器"""
    if func_name == "get_app_state":
        return clean_ax_text(raw_text, max_chars=4800)
    elif func_name == "list_apps":
        return format_app_list(raw_text)
    else:
        if len(raw_text) > 500:
            return raw_text[:500] + "..."
        return raw_text if raw_text.strip() else "ok"


def launch_mac_app(app_name: str, bundle_id: str = "") -> str:
    """原生极速启动并置顶激活 macOS 应用程序"""
    name_map = {
        "计算器": "Calculator",
        "备忘录": "Notes",
        "时钟": "Clock",
        "计时器": "Clock",
        "闹钟": "Clock",
        "秒表": "Clock",
        "clock": "Clock",
        "日历": "Calendar",
        "地图": "Maps",
        "音乐": "Music",
        "网易云音乐": "NeteaseMusic",
        "播客": "Podcasts",
        "微信": "WeChat",
        "飞书": "Feishu",
        "邮件": "Mail",
        "终端": "Terminal",
        "访达": "Finder",
        "相册": "Photos",
        "照片": "Photos",
        "提醒事项": "Reminders",
        "系统设置": "System Settings",
        "设置": "System Settings",
        "outlook": "Microsoft Outlook",
        "microsoft outlook": "Microsoft Outlook",
        "word": "Microsoft Word",
        "microsoft word": "Microsoft Word",
        "chrome": "Google Chrome",
        "google chrome": "Google Chrome",
        "谷歌浏览器": "Google Chrome",
        "safari": "Safari",
        "discord": "Discord",
        "keynote": "Keynote",
        "pages": "Pages",
        "numbers": "Numbers",
        "文本编辑": "TextEdit",
        "活动监视器": "Activity Monitor",
        "系统信息": "System Information",
        "qq": "QQ",
        "腾讯qq": "QQ",
        "cursor": "Cursor",
        "obsidian": "Obsidian",
        "libreoffice": "LibreOffice",
    }
    bid_map = {
        "calculator": "com.apple.calculator",
        "计算器": "com.apple.calculator",
        "notes": "com.apple.Notes",
        "备忘录": "com.apple.Notes",
        "clock": "com.apple.clock",
        "时钟": "com.apple.clock",
        "计时器": "com.apple.clock",
        "闹钟": "com.apple.clock",
        "秒表": "com.apple.clock",
        "safari": "com.apple.Safari",
        "chrome": "com.google.Chrome",
        "google chrome": "com.google.Chrome",
        "谷歌浏览器": "com.google.Chrome",
        "calendar": "com.apple.iCal",
        "日历": "com.apple.iCal",
        "wechat": "com.tencent.xinWeChat",
        "微信": "com.tencent.xinWeChat",
        "qq": "com.tencent.qq",
        "腾讯qq": "com.tencent.qq",
        "feishu": "com.electron.lark",
        "飞书": "com.electron.lark",
        "mail": "com.apple.mail",
        "邮件": "com.apple.mail",
        "outlook": "com.microsoft.Outlook",
        "microsoft outlook": "com.microsoft.Outlook",
        "word": "com.microsoft.Word",
        "microsoft word": "com.microsoft.Word",
        "discord": "com.hnc.Discord",
        "keynote": "com.apple.iWork.Keynote",
        "pages": "com.apple.iWork.Pages",
        "numbers": "com.apple.iWork.Numbers",
        "textedit": "com.apple.TextEdit",
        "文本编辑": "com.apple.TextEdit",
        "cursor": "com.todesktop.230313mzl4w4u92",
        "obsidian": "md.obsidian",
        "libreoffice": "org.libreoffice.script",
    }

    target_bid = bundle_id or bid_map.get(app_name.lower())
    if target_bid:
        res = subprocess.run(["open", "-b", target_bid], capture_output=True, text=True)
        if res.returncode == 0:
            try:
                subprocess.run(["osascript", "-e", f'tell application id "{target_bid}" to activate'], capture_output=True, timeout=2)
            except Exception:
                pass
            return f"成功打开并激活应用: {app_name} (bundle_id: {target_bid})"

    target_name = name_map.get(app_name, app_name)
    res = subprocess.run(["open", "-a", target_name], capture_output=True, text=True)
    if res.returncode == 0:
        try:
            subprocess.run(["osascript", "-e", f'tell application "{target_name}" to activate'], capture_output=True, timeout=2)
        except Exception:
            pass
        return f"成功打开并激活应用: {target_name}"

    res2 = subprocess.run(["open", "-a", app_name], capture_output=True, text=True)
    if res2.returncode == 0:
        try:
            subprocess.run(["osascript", "-e", f'tell application "{app_name}" to activate'], capture_output=True, timeout=2)
        except Exception:
            pass
        return f"成功打开并激活应用: {app_name}"

    return f"未能打开应用 '{app_name}': {res.stderr or res2.stderr or '未找到对应应用程序'}"


BROWSER_TOOLS = {decl.name for decl in get_browser_function_declarations()}

CUSTOM_ASR_VOCABULARY = [
    "长鑫科技", "IT之家", "kimi-cu", "Safari", "Chrome", "Google Chrome",
    "计算器", "备忘录", "访达", "Finder", "微信", "终端", "Terminal",
    "百度", "哔哩哔哩", "Ego Lite", "音量", "窗口", "标签页",
]


def build_gemini_function_declarations(
    mcp_tools: List[Any],
    is_extended_thinking: bool = False,
) -> List[types.FunctionDeclaration]:
    """
    根据 Gemini 3.8 官方规范构建 FunctionDeclaration 并显式声明 behavior:
    - Extended Thinking 模型：官方强制要求所有工具必须为 behavior="NON_BLOCKING"
    - 3.8-live 极速模型：macOS 桌面物理操作与浏览器交互显式声明 behavior="BLOCKING"，保证严格串行
    """
    target_behavior = (
        types.Behavior.NON_BLOCKING if is_extended_thinking else types.Behavior.BLOCKING
    )
    gemini_functions = [
        types.FunctionDeclaration(
            name=t.name,
            description=t.description or "",
            parameters=t.input_schema or {"type": "object", "properties": {}},
            behavior=target_behavior,
        )
        for t in mcp_tools
    ]

    open_app_tool = types.FunctionDeclaration(
        name="open_app",
        description="在 macOS 上启动或前台激活任何应用程序（例如 计算器, 备忘录, 音乐, 微信, Safari, Chrome, 日历等）。如果应用未运行会自动启动，若已在运行则直接置顶激活到前台。用户说'打开XXX'时优先使用此工具。",
        parameters={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "应用程序名称，支持中文或英文，例如 '计算器', 'Calculator', '备忘录', 'Notes', 'Safari', '微信' 等"
                },
                "bundle_id": {
                    "type": "string",
                    "description": "可选。应用的 Bundle Identifier，例如 'com.apple.calculator', 'com.apple.Safari' 等"
                }
            },
            "required": ["name"]
        },
        behavior=target_behavior,
    )
    gemini_functions.append(open_app_tool)
    gemini_functions.extend(get_browser_function_declarations(behavior=target_behavior.value))
    return gemini_functions


class ToolExecutor:
    """以 ToolResultContract 为 seam，分发 native-app / kimi-cu / Ego Lite 三个 adapter。"""

    def __init__(
        self,
        mcp_session=None,
        launch_app: Optional[Callable[..., str]] = None,
    ):
        self.mcp_session = mcp_session
        self.launch_app = launch_app or launch_mac_app

    def apply_defaults(self, func_name: str, func_args: Dict[str, Any]) -> Dict[str, Any]:
        args = dict(func_args or {})
        if func_name in ["press_key", "type_text", "click"]:
            if not args.get("app") and not args.get("pid"):
                args["app"] = "com.apple.finder"
        if func_name in ["press_key", "type_text"]:
            args.setdefault("activate", True)
        elif func_name == "get_app_state":
            args.setdefault("mode", "ax")
        return args

    async def execute(self, func_name: str, func_args: Dict[str, Any]) -> ToolResultContract:
        args = self.apply_defaults(func_name, func_args)
        try:
            if func_name == "open_app":
                return await self._native_app(args)
            if func_name in BROWSER_TOOLS:
                return await self._ego_lite(func_name, args)
            return await self._kimi_cu(func_name, args)
        except asyncio.CancelledError:
            raise
        except Exception as err:
            return ToolResultContract(
                ok=False,
                action=func_name,
                status="error",
                summary="执行异常",
                error=str(err),
            )

    async def _native_app(self, func_args: Dict[str, Any]) -> ToolResultContract:
        target_app = func_args.get("name", "") or func_args.get("app", "")
        target_bid = func_args.get("bundle_id", "")
        raw_text = await asyncio.to_thread(self.launch_app, target_app, target_bid)
        ok = raw_text.startswith("成功打开并激活应用")
        return ToolResultContract(
            ok=ok,
            action="open_app",
            status="success" if ok else "error",
            summary=raw_text,
            side_effects="app_launched" if ok else "none",
            error=None if ok else raw_text,
        )

    async def _ego_lite(self, func_name: str, func_args: Dict[str, Any]) -> ToolResultContract:
        if func_name == "browser_open":
            url = func_args.get("url", "") or func_args.get("url_or_kw", "")
            res_text = await browser_open(url)
            err = is_browser_error(res_text)
            return ToolResultContract(
                ok=not err,
                action="browser_open",
                status="success" if not err else "error",
                summary="已打开网页" if not err else res_text[:60],
                data=res_text,
                side_effects="navigation" if not err else "none",
                error=res_text if err else None,
            )
        if func_name == "browser_search":
            query = func_args.get("query", "")
            engine = func_args.get("engine", "baidu")
            res_text = await browser_search(query, engine)
            err = is_browser_error(res_text)
            return ToolResultContract(
                ok=not err,
                action="browser_search",
                status="success" if not err else "error",
                summary=f"已搜索关键词 '{query}'" if not err else res_text[:60],
                data=res_text,
                side_effects="navigation" if not err else "none",
                error=res_text if err else None,
            )
        if func_name == "browser_get_content":
            res_text = await browser_get_content()
            err = is_browser_error(res_text)
            return ToolResultContract(
                ok=not err,
                action="browser_get_content",
                status="success" if not err else "error",
                summary="已提取页面内容" if not err else res_text[:60],
                data=res_text,
                error=res_text if err else None,
            )
        if func_name == "browser_list_actions":
            max_items = func_args.get("max_items", 25)
            res_text = await browser_list_actions(max_items=max_items)
            err = is_browser_error(res_text)
            return ToolResultContract(
                ok=not err,
                action="browser_list_actions",
                status="success" if not err else "error",
                summary="已获取页面可操作候选项" if not err else res_text[:60],
                data=res_text,
                error=res_text if err else None,
            )
        if func_name == "browser_click":
            target = (
                func_args.get("text", "")
                or func_args.get("text_or_selector", "")
                or func_args.get("target", "")
            )
            res_text = await browser_click(target)
            err = is_browser_error(res_text)
            return ToolResultContract(
                ok=not err,
                action="browser_click",
                status="success" if not err else "error",
                summary=res_text[:80],
                side_effects="ui_updated" if not err else "none",
                error=res_text if err else None,
            )
        if func_name == "browser_close":
            close_win = func_args.get("close_window", True)
            res_text = await browser_close(close_window=close_win)
            return ToolResultContract(
                ok=True,
                action="browser_close",
                status="success",
                summary="已成功关闭浏览器页面并退出窗口",
                data=res_text,
                side_effects="window_closed",
            )
        direction = func_args.get("direction", "down")
        res_text = await browser_scroll(direction)
        err = is_browser_error(res_text)
        return ToolResultContract(
            ok=not err,
            action="browser_scroll",
            status="success" if not err else "error",
            summary=res_text[:80],
            side_effects="ui_updated" if not err else "none",
            error=res_text if err else None,
        )

    async def _kimi_cu(self, func_name: str, func_args: Dict[str, Any]) -> ToolResultContract:
        try:
            mcp_res = await asyncio.wait_for(
                self.mcp_session.call_tool(func_name, func_args),
                timeout=12.0
            )
        except asyncio.TimeoutError:
            return ToolResultContract(
                ok=False,
                action=func_name,
                status="error",
                summary=f"{func_name} 执行超时 (12秒无响应)",
                error=f"kimi-cu 工具 {func_name} 执行超时，目标控件可能未就绪或未找到",
            )
        except Exception as err:
            return ToolResultContract(
                ok=False,
                action=func_name,
                status="error",
                summary=f"{func_name} 执行异常",
                error=str(err),
            )
        is_err = bool(getattr(mcp_res, "is_error", False) or getattr(mcp_res, "isError", False))
        texts = []
        for item in mcp_res.content:
            if hasattr(item, "text") and item.text:
                texts.append(item.text)
            elif hasattr(item, "data"):
                texts.append("[截图像素数据已捕获]")
        raw_text = "\n".join(texts) if texts else ("MCP 执行失败" if is_err else "ok")
        cleaned_text = format_tool_result(func_name, raw_text)
        return ToolResultContract(
            ok=not is_err,
            action=func_name,
            status="success" if not is_err else "error",
            summary=f"已执行 {func_name}" if not is_err else f"{func_name} 执行失败",
            data=cleaned_text,
            side_effects="ui_updated" if not is_err else "none",
            error=cleaned_text if is_err else None,
        )


# 自动配置本地代理端口
if not os.environ.get("http_proxy") and not os.environ.get("https_proxy"):
    for port in [7890, 7897, 10808]:
        try:
            import socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.1)
                if s.connect_ex(("127.0.0.1", port)) == 0:
                    proxy_url = f"http://127.0.0.1:{port}"
                    os.environ["http_proxy"] = proxy_url
                    os.environ["https_proxy"] = proxy_url
                    os.environ.pop("all_proxy", None)
                    break
        except Exception:
            pass


MODEL_FAST = "gemini-3.8-live"
MODEL_THINKING = "gemini-3.8-live-extended-thinking"

SYSTEM_INSTRUCTION = """
【重要：绝对语言约束（HIGHEST PRIORITY）】：
1. 你必须无条件、始终使用清晰流利的【中文】与用户进行全流程交流！
2. 严禁用英文或其它外语开场或打招呼！首次打招呼必须使用中文（例如：“您好！我是您的 macOS 实时语音电脑管家，请问有什么可以帮您？”）。
3. 无论用户当前使用什么语言输入、无论网页和工具返回什么语言，你的口语回复、思考总结、状态反馈必须全部使用纯正自然的中文！

【普通话语音识别对齐与领域纠错规则（CRITICAL FOR SPEECH RECOGNITION）】：
1. 用户输入为中文普通话实时语音流。在声学与语义解码时，必须强制结合科技、macOS 桌面操作与互联网资讯语境进行纠偏，绝对严禁出现跨语言幻觉（严禁输出印地语、梵文、法语、西班牙语等无关外语）！
2. 常见易混淆科技、桌面专有名词与日常口语严格对照表（结合发音、声调与上下文精准对齐）：
   - “我让你帮我搜” / “让你帮我搜” / “请帮我搜”（绝对不可误听或脑补为“玩你帮我搜”！）。
   - “打开了吗” / “打开了么”（绝对不可误听为“打卡吗”或“考勤打卡”！）。
   - “十月一号” / “10月1号”（绝对不可丢失开头“十”或“10”而误听为“是一号”！）。
   - “长鑫科技” / “长信科技” / “长江存储” / “中芯国际” 等半导体与科技公司（发音常有送气音 ch/zh，绝对不可误听或脑补为“朱朝阳日记”、“朝阳日记”等非科技词汇！）。
   - “浏览器”（Ego Lite 浏览器、Safari、Chrome 等，发音为 liú lǎn qì，绝对不可误听为“暖气”！）。
   - “IT之家”（中国知名科技数码资讯网站 ithome.com，发音为 IT zhī jiā，绝对不可误听为“IT职教”或其它学校！）。
   - “计算器”（发音 jì suàn qì，不可误听为“光盘录”或“计时器”）。
   - “Discord”（流行通讯软件，不可误听为“disc code”）。
   - “Outlook” / “邮箱” / “邮件”。
   - “模型” / “大模型” / “AI模型”（发音 mó xíng，绝对不可误听为“魔鬼”！）。
3. 遇到发音微弱、送气音轻或同音字时，强制结合当前桌面操作上下文推断为用户真实意图或桌面软件名称。

【核心身份与上下文记忆原则】：
1. 你具备跨轮次的上下文记忆能力！你能够清晰记住用户在前面各轮说过的指令、参数、偏好以及上一步打开的应用与网页。
2. 当用户提及“刚才”、“之前”、“上一个”等指代词时，你必须根据前序对话历史做出准确衔接与回应！

你是一个运行在 macOS 上的实时语音电脑操作管家。
你能操控 macOS 桌面上的所有应用和窗口，并具备强劲的网页浏览与检索能力。

【工具能力与使用规则】：
1. 应用程序管理（macOS 本地）：
   - open_app: 打开或前台激活任意应用程序（计算器、时钟/计时器、备忘录、微信、QQ、Outlook、Word、音乐、Pages 等）。用户要求“打开XXX软件”时必须优先调用此工具！
   - list_apps: 查看当前正在运行的所有应用程序（包含 name, bundle_id, pid）。操作前可用此工具确认目标应用是否在运行。
   - get_app_state: 获取指定桌面应用的窗口控件树（参数 app 传 bundle_id，mode 建议使用 'ax' 获取极速控件树）。
   - click: 点击目标桌面应用的控件（传 app 和控件 index，或点击坐标 x, y）。
   - type_text: 向目标应用输入文本（优先传 index 获得焦点，支持 clear=True 清空后输入，submit=True 按回车提交。若要使用 x, y 坐标，必须先调用 get_app_state 获取截图）。
   - press_key: 向应用发送按键或快捷键（如 return, enter, escape, cmd+c, cmd+v, cmd+w, cmd+n 等，参数 app 传 bundle_id）。
   - scroll: 滚动桌面应用窗口。

2. 网页浏览与联网搜索（基于 Ego Lite 极速浏览器）：
   - browser_open: 打开指定网址或 URL，返回网页标题与精简正文（如 browser_open(url="https://www.ithome.com")）。
   - browser_search: 在浏览器中搜索关键词并提炼要点（如 browser_search(query="长鑫科技 最新报道")）。
   - browser_get_content: 抓取当前已打开网页的正文内容并总结。
   - browser_list_actions: 列出当前网页中所有可交互操作元素（链接、按钮）及稳定编号 [#ID]。当页面选项较多或可能出现重复文案误点击时，先调用此工具列出候选编号！
   - browser_click: 在当前网页中点击指定链接或按钮（支持传入候选编号如 "#1" 或标题文字）。
   - browser_scroll: 在当前网页中向上或向下滚动（如 browser_scroll(direction="down")）。
   【出行预订与指定日期页面纪律】：当用户查询特定日期（如“10月1日”、“国庆节”、“明天”等）的机票、酒店、火车票时，若直接打开链接，必须核对页面当前显示的出发日期是否与用户要求严格一致！如果页面停留为默认日期（如今天），必须调用 browser_list_actions 找到日期选项并点击目标日期，或调用 browser_search(query="天津到西安机票 10月1日 价格 携程") 直接查询，绝不可对未切中日期的页面谎称已查好！
   【绝对纪律】：严禁擅自打开 Safari 或 Google Chrome！所有网页访问、互联网资讯搜索、网页内容阅读一律必须且只能在 Ego Lite 浏览器（通过 browser_* 系列工具）中完成。若在当前网页中未找到目标，调用 browser_search 或 browser_scroll 继续寻找，绝对不允许调用 open_app("Safari")！

【安全防护与执行纪律】：
1. 涉及永久删除文件、系统关机、恶意脚本执行等高危命令一律被安全策略拦截；
2. 工具结果返回结构化状态，若被安全策略拦截或被用户打断，需向用户如实说明。

【交互与口语原则】：
1. 【静默动作，一次性总结汇报】：
   - 执行操作时直接下发工具，不要在调用前说废话；
   - 动作执行成功后，用简明自然的中文口语告知用户最终结果；
   - 严禁对同一动作连续死循环重复调用！若工具已成功返回，立即结束动作并作口语回复。
2. 【日常问答自然连贯】：
   - 用户日常打招呼或闲聊时，用纯正中文自然流畅地回答。
"""

# 音频参数
MIC_RATE = 16000     # 录音采样率 (16kHz, int16 单声道)
SPK_RATE = 24000     # 播音采样率 (24kHz, int16 单声道)
CHUNK_SIZE = 1024    # 64ms 块

def find_audio_devices(preferred_mic=None):
    """
    智能查找麦克风：
    1. 若显式指定了 preferred_mic，优先按关键词搜索指定设备；
    2. 若系统检测到高品质无线领夹麦 (Wireless Mic Rx)，默认首选以获得最优近场拾音效果；
    3. 若未连接无线麦，则自动回退至 macOS 系统当前活跃的默认输入设备 (sd.default.device[0])；
    4. 自动探测并返回最优声道数（单声道或双声道）。
    """
    devices = sd.query_devices()
    mic_idx = None
    mic_name = "系统默认麦克风"
    channels = 1

    # 1. 优先匹配用户显式指定的麦克风
    if preferred_mic and str(preferred_mic).strip():
        pref = str(preferred_mic).strip().lower()
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                if pref in d.get("name", "").lower():
                    mic_idx = idx
                    mic_name = d["name"]
                    channels = min(2, d.get("max_input_channels", 1))
                    break

    # 2. 默认高优先级：若检测到已连接的高品质无线领夹麦 (Wireless Mic)，优先使用以保障专业级近场音质
    if mic_idx is None:
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0 and "wireless mic" in d.get("name", "").lower():
                mic_idx = idx
                mic_name = d["name"]
                channels = min(2, d.get("max_input_channels", 1))
                break

    # 3. 未检测到无线麦时，回退使用 macOS 当前默认输入设备 (sd.default.device[0])
    if mic_idx is None:
        try:
            default_in = sd.default.device[0]
            if default_in is not None and default_in >= 0 and default_in < len(devices):
                d = devices[default_in]
                if d.get("max_input_channels", 0) > 0:
                    mic_idx = default_in
                    mic_name = d["name"]
                    channels = min(2, d.get("max_input_channels", 1))
        except Exception:
            pass

    # 3. 若系统默认设备不可用，回退选取首个可用输入设备
    if mic_idx is None:
        for idx, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                mic_idx = idx
                mic_name = d["name"]
                channels = min(2, d.get("max_input_channels", 1))
                break

    if mic_idx is None:
        mic_idx = 0
        mic_name = devices[0]["name"] if devices else "默认麦克风"
        channels = 1

    return mic_idx, mic_name, channels


class SmoothAudioPlayer:
    """带 Jitter Buffer 预缓冲平滑输出的音频播放器，彻底消除顿挫与断续"""
    def __init__(self, sample_rate=SPK_RATE, max_queue_size=150):
        self.sample_rate = sample_rate
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.running = True
        self.is_playing = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        try:
            with sd.RawOutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=CHUNK_SIZE
            ) as stream:
                while self.running:
                    try:
                        # 1. 空闲启动时，真实预缓冲（积攒 2-3 块或最多等 100ms），吸收网络抖动
                        first_chunks = []
                        start_wait = time.time()
                        while len(first_chunks) < 3 and (time.time() - start_wait < 0.1):
                            try:
                                c = self.queue.get(timeout=0.03)
                                if c is None:
                                    return
                                first_chunks.append(c)
                            except queue.Empty:
                                if first_chunks:
                                    break

                        if not first_chunks:
                            self.is_playing = False
                            continue

                        self.is_playing = True
                        for c in first_chunks:
                            stream.write(c)

                        # 2. 连续播放循环
                        while self.running:
                            try:
                                chunk = self.queue.get(timeout=0.15)
                                if chunk is None:
                                    return
                                stream.write(chunk)
                            except queue.Empty:
                                self.is_playing = False
                                break
                    except queue.Empty:
                        self.is_playing = False
                        continue
        except Exception as e:
            log_event("AUDIO_ERROR", str(e))

    def write(self, data: bytes):
        if self.running and data:
            try:
                self.queue.put_nowait(data)
            except queue.Full:
                # 队列溢出时主动丢弃最老一帧，吸收积压
                try:
                    self.queue.get_nowait()
                except Exception:
                    pass
                try:
                    self.queue.put_nowait(data)
                except Exception:
                    pass

    def is_busy(self):
        return self.is_playing or not self.queue.empty()

    def interrupt(self):
        """立即清空音频缓冲区并停止当前播放"""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        self.is_playing = False

    def stop(self):
        self.running = False
        try:
            self.queue.put_nowait(None)
        except Exception:
            pass
        self.thread.join(timeout=0.5)


async def run_session(
    api_key,
    selected_model,
    voice_name,
    mic_idx,
    mic_name,
    mic_channels,
    mcp_session,
    gemini_functions,
    player,
    shutdown_event,
    memory: ConversationMemory,
    user_threshold=None,
    user_interrupt_threshold=None,
    strict_policy=True,
    session_state=None
):
    """
    单个全双工实时会话生命周期
    - 维持常驻 WebSocket 连接，采用官方公开 session.receive() 与 session_resumption
    - HybridVAD 模式：连续流式 send_realtime_input + 本地近场门控 + audio_stream_end
    - 本地对话记忆池 (ConversationMemory) 支持多轮累积与断线回灌
    - 带 CancellationToken 的可取消多步执行与全链路打断
    - 安全执行层 (Tool Policy) 策略审查、调用预算、去重拦截与结构化结果契约
    """
    client = genai.Client(api_key=api_key)
    policy_manager = ToolPolicyManager(strict_mode=strict_policy)
    policy_manager.reset_turn()

    if session_state is None:
        session_state = {"handle": None}

    thinking_config = None
    if "thinking" in selected_model:
        level = os.environ.get("GEMINI_THINKING_LEVEL", "low").lower()
        thinking_config = types.ThinkingConfig(include_thoughts=True, thinking_level=level)

    last_resumption_handle = session_state.get("handle")
    session_resumption_cfg = (
        types.SessionResumptionConfig(handle=last_resumption_handle)
        if last_resumption_handle
        else types.SessionResumptionConfig()
    )
    target_tokens = int(os.environ.get("CONTEXT_TARGET_TOKENS", 32000))
    context_compression_cfg = types.ContextWindowCompressionConfig(
        sliding_window=types.SlidingWindow(target_tokens=target_tokens)
    )

    realtime_input_cfg = types.RealtimeInputConfig(
        automatic_activity_detection=types.AutomaticActivityDetection(
            silence_duration_ms=1000,
            prefix_padding_ms=100,
            start_of_speech_sensitivity=types.StartSensitivity.START_SENSITIVITY_LOW,
            end_of_speech_sensitivity=types.EndSensitivity.END_SENSITIVITY_LOW,
        ),
        activity_handling=types.ActivityHandling.START_OF_ACTIVITY_INTERRUPTS,
    )

    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        system_instruction=types.Content(
            parts=[types.Part.from_text(text=SYSTEM_INSTRUCTION)]
        ),
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice_name
                )
            )
        ),
        input_audio_transcription=types.AudioTranscriptionConfig(
            custom_vocabulary=CUSTOM_ASR_VOCABULARY
        ),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        thinking_config=thinking_config,
        tools=[types.Tool(function_declarations=gemini_functions)],
        session_resumption=session_resumption_cfg,
        context_window_compression=context_compression_cfg,
        realtime_input_config=realtime_input_cfg,
    )

    loop = asyncio.get_running_loop()

    audio_turn_queue = AudioTurnQueue(maxsize=250)
    reconnect_event = asyncio.Event()

    # 近场降噪门限参数 (针对 Wireless Mic 专业领夹麦进行高灵敏起呼，杜绝任何开头漏字)
    START_THRESHOLD = user_threshold or 65
    HOLD_THRESHOLD = max(30, min(45, int(START_THRESHOLD * 0.45)))
    ATTACK_FRAMES = 1  # 1 帧检测即响应，配合 1280ms 前滚缓冲确保首字辅音 100% 完整推入
    SILENCE_CHUNKS = 13  # 约 832 毫秒断句窗口，与服务端 1000ms 自动 VAD 形成快慢双通道 Hybrid VAD
    MIN_PEAK_RMS = max(80, int(START_THRESHOLD * 1.2))

    # 分级打断门限 (Barge-in Threshold Hierarchy)
    # 1. 播报打断门限 (需近场清晰发声，拦截扬声器漏音误打断；尾音阶段自动回落)
    INTERRUPT_SPEAKING_RMS = user_interrupt_threshold or max(260, int(START_THRESHOLD * 2.8))
    # 2. 动作执行打断门限 (高代价桌面物理操作，提升打断门槛，需近场明确喊停)
    INTERRUPT_EXECUTING_RMS = max(380, int(START_THRESHOLD * 3.5)) if not user_interrupt_threshold else max(int(user_interrupt_threshold * 1.25), 380)
    # 3. 思考打断门限 (Gemini 思考时扬声器静音，近场开口即打断，杜绝吞字)
    INTERRUPT_THINKING_RMS = max(160, int(START_THRESHOLD * 2.0)) if not user_interrupt_threshold else max(user_interrupt_threshold, 160)

    pre_roll = collections.deque(maxlen=20)  # 约 1280ms 常驻滚动前滚缓冲，100% 留存整句话前摇、吸气声与首字辅音
    is_speaking = False
    speech_frames = 0
    speech_end_time = 0.0
    attack_count = 0
    silence_count = 0
    interrupt_frames = 0
    is_ready_to_listen = False
    calib_samples = []
    last_cli_print_time = 0.0

    turn_ctrl = TurnController(
        use_interaction_status="thinking" in selected_model,
        policy_manager=policy_manager
    )
    tool_executor = ToolExecutor(mcp_session=mcp_session)
    send_lock = asyncio.Lock()

    def handle_voice_barge_in(reason: str, initial_chunks: Optional[List[bytes]] = None):
        """在事件循环主线程中线程安全地处理打断、清理队列与开启新轮次，完整注入打断前摇音频"""
        turn_ctrl.interrupt(reason)
        audio_turn_queue.discard_turn()
        turn_ctrl.new_turn(f"speech_after_{reason}")
        if initial_chunks:
            for chunk in initial_chunks:
                audio_turn_queue.enqueue_audio(chunk)

    def handle_voice_speech_start(chunks_to_send: list):
        """在事件循环主线程中线程安全地开启新轮次并推送起呼前摇音频"""
        turn_ctrl.new_turn("speech_start")
        for pr_chunk in chunks_to_send:
            audio_turn_queue.enqueue_audio(pr_chunk)

    def mic_callback(indata, frames, time_info, status):
        nonlocal is_speaking, speech_frames, speech_end_time, attack_count, silence_count, interrupt_frames
        nonlocal START_THRESHOLD, HOLD_THRESHOLD, MIN_PEAK_RMS
        nonlocal INTERRUPT_SPEAKING_RMS, INTERRUPT_EXECUTING_RMS, INTERRUPT_THINKING_RMS
        nonlocal last_cli_print_time

        if mic_channels == 2:
            stereo = np.frombuffer(indata, dtype=np.int16).reshape(-1, 2)
            # 关键：Wireless Mic Rx 为专业双通道硬件，固定提取 Channel 0 主声道，彻底根除逐帧跳切带来的波形抖动
            mono_samples = stereo[:, 0]
        else:
            mono_samples = np.frombuffer(indata, dtype=np.int16)

        # 硬件直流偏移去除 (DC Offset Removal)：消除 -6.95 的固定硬件偏置，保留纯净自然的声学基频
        mono_f = mono_samples.astype(np.float32)
        offset = float(np.mean(mono_f))
        if abs(offset) > 0.5:
            mono_f = mono_f - offset
            mono_samples = np.clip(mono_f, -32767, 32767).astype(np.int16)

        rms = int(np.sqrt(np.mean(mono_f**2)))
        raw_bytes = mono_samples.tobytes()

        # 关键机制：无条件持续向滚动前摇缓冲压入新鲜帧，确保任何时刻起呼或打断均能无损获取 1280ms 前摇
        pre_roll.append(raw_bytes)

        if not is_ready_to_listen:
            calib_samples.append(rms)
            return

        now = time.time()
        should_print_cli = (now - last_cli_print_time >= 0.1)
        bars = "▇" * min(12, rms // 30) + "░" * max(0, 12 - rms // 30)

        # 1. 扬声器播报中：检测打断 (需近场明确发声达到 INTERRUPT_SPEAKING_RMS 且持续2帧，彻底拦截扬声器回音自激)
        if turn_ctrl.state == STATE_SPEAKING:
            if rms >= INTERRUPT_SPEAKING_RMS:
                interrupt_frames += 1
                if interrupt_frames >= 2:
                    player.interrupt()
                    chunks = list(pre_roll)
                    loop.call_soon_threadsafe(handle_voice_barge_in, "speaking", chunks)
                    is_speaking = True
                    speech_frames = 1
                    attack_count = 0
                    silence_count = 0
                    interrupt_frames = 0
                    log_event("USER_INTERRUPT", f"User interrupted speaking (rms={rms}, threshold={INTERRUPT_SPEAKING_RMS})")
                    sys.stdout.write(f"\r🛑 [\033[1;31m已打断播报，请继续说...\033[0m]                           \n")
                    sys.stdout.flush()
                    last_cli_print_time = now
                    return
            else:
                interrupt_frames = 0
            if should_print_cli:
                sys.stdout.write(f"\r🗣️ [\033[1;35mGemini 正在播报...\033[0m] 音量: |{bars}| ({rms:3d}/{INTERRUPT_SPEAKING_RMS}) ")
                sys.stdout.flush()
                last_cli_print_time = now
            return

        # 2. kimi-cu 正在执行动作：物理桌面操作打断门槛极高，非近场大声明确喊话不中断
        if turn_ctrl.state == STATE_EXECUTING:
            if rms >= INTERRUPT_EXECUTING_RMS:
                interrupt_frames += 1
                if interrupt_frames >= 3:
                    player.interrupt()
                    chunks = list(pre_roll)
                    loop.call_soon_threadsafe(handle_voice_barge_in, "executing", chunks)
                    is_speaking = True
                    speech_frames = 1
                    attack_count = 0
                    silence_count = 0
                    interrupt_frames = 0
                    log_event("USER_INTERRUPT", f"User interrupted executing state (rms={rms}, threshold={INTERRUPT_EXECUTING_RMS})")
                    sys.stdout.write(f"\r🛑 [\033[1;31m已打断动作执行，请继续说...\033[0m]                           \n")
                    sys.stdout.flush()
                    last_cli_print_time = now
                    return
            else:
                interrupt_frames = 0
            if should_print_cli:
                sys.stdout.write(f"\r⚙️ [\033[1;33mkimi-cu 正在执行桌面动作...\033[0m] 音量: |{bars}| ({rms:3d}/{INTERRUPT_EXECUTING_RMS}) ")
                sys.stdout.flush()
                last_cli_print_time = now
            return

        # 3. 正在思考中 (支持气口续说自然合并；真正打断需达 INTERRUPT_THINKING_RMS 且持续2帧，杜绝微小环境杂音误打断)
        if turn_ctrl.state == STATE_THINKING:
            time_since_speech_end = now - speech_end_time
            if time_since_speech_end < 0.85 and not turn_ctrl.has_active_tool and not player.is_busy():
                if rms >= START_THRESHOLD:
                    # 视为同一轮长句的气口自然续接，直接切回 LISTENING，追加音频流，绝不清空队列
                    loop.call_soon_threadsafe(turn_ctrl.set_state, STATE_LISTENING)
                    is_speaking = True
                    silence_count = 0
                    speech_frames += 1
                    log_event("SPEECH_CONTINUE", f"Merged speech continuation during thinking (rms={rms}, gap={time_since_speech_end:.2f}s)")
                    chunks = list(pre_roll)
                    for c in chunks[-4:]:
                        loop.call_soon_threadsafe(audio_turn_queue.enqueue_audio, c)
                    return
            else:
                if rms >= INTERRUPT_THINKING_RMS:
                    interrupt_frames += 1
                    if interrupt_frames >= 2:
                        player.interrupt()
                        chunks = list(pre_roll)
                        loop.call_soon_threadsafe(handle_voice_barge_in, "thinking", chunks)
                        is_speaking = True
                        speech_frames = 1
                        attack_count = 0
                        silence_count = 0
                        interrupt_frames = 0
                        log_event("USER_INTERRUPT", f"User spoke during thinking, transitioned to new speech (rms={rms}, threshold={INTERRUPT_THINKING_RMS})")
                        sys.stdout.write(f"\r🎤 [\033[1;32m检测到打断并重新说话，实时响应中...\033[0m]                           \n")
                        sys.stdout.flush()
                        last_cli_print_time = now
                        return
                else:
                    interrupt_frames = 0

            if should_print_cli:
                sys.stdout.write(f"\r🧠 [\033[1;36mGemini 正在处理中...\033[0m] 音量: |{bars}| ({rms:3d}/{INTERRUPT_THINKING_RMS}) ")
                sys.stdout.flush()
                last_cli_print_time = now
            return

        # 4. 空闲监听中 (HybridVAD 连续流式输入，零延迟极速响应 + 1280ms 全量前摇推送)
        if not is_speaking:
            if rms >= START_THRESHOLD:
                is_speaking = True
                speech_frames = 1
                silence_count = 0
                attack_count = 0
                chunks_to_send = list(pre_roll)
                loop.call_soon_threadsafe(handle_voice_speech_start, chunks_to_send)
                log_event("USER_SPEECH_START", f"Streaming speech started (rms={rms})")
                if should_print_cli:
                    sys.stdout.write(f"\r🎤 [\033[1;32m正在流式说话\033[0m] 音量: |{bars}| ({rms:3d}) ")
                    sys.stdout.flush()
                    last_cli_print_time = now
            else:
                if should_print_cli:
                    if rms >= max(35, int(START_THRESHOLD * 0.6)):
                        sys.stdout.write(f"\r🎤 [\033[93m收音中\033[0m] 音量: |{bars}| ({rms:3d}/{START_THRESHOLD}) ")
                    else:
                        sys.stdout.write(f"\r🎤 [\033[90m监听中\033[0m] 音量: |{bars}| ({rms:3d}/{START_THRESHOLD}) ")
                    sys.stdout.flush()
                    last_cli_print_time = now
        else:
            # 持续流式推送 PCM 块
            speech_frames += 1
            loop.call_soon_threadsafe(audio_turn_queue.enqueue_audio, raw_bytes)
            if rms >= HOLD_THRESHOLD:
                silence_count = 0
                if should_print_cli:
                    sys.stdout.write(f"\r🎤 [\033[1;32m正在流式说话\033[0m] 音量: |{bars}| ({rms:3d}) ")
                    sys.stdout.flush()
                    last_cli_print_time = now
            else:
                silence_count += 1
                # 动态断句容忍：刚起呼阶段（发声小于 20 帧约 1.2 秒），轻微放宽停顿容忍到 16 帧（约 1.0 秒，快于服务端兜底）
                required_silence = 16 if speech_frames < 20 else SILENCE_CHUNKS
                if should_print_cli:
                    sys.stdout.write(f"\r🎤 [\033[1;36m检测停顿 {silence_count}/{required_silence}\033[0m] 音量: |{bars}| ({rms:3d}) ")
                    sys.stdout.flush()
                    last_cli_print_time = now

                if silence_count >= required_silence:
                    is_speaking = False
                    silence_count = 0
                    attack_count = 0
                    speech_frames = 0
                    speech_end_time = now
                    # 发送音频流结束标记，通知 Gemini 即刻开始响应
                    loop.call_soon_threadsafe(audio_turn_queue.finish_turn)
                    loop.call_soon_threadsafe(turn_ctrl.set_state, STATE_THINKING)
                    log_event("USER_SPEECH_END", "Speech ended, queued AUDIO_STREAM_END")
                    sys.stdout.write(f"\r⚡ [\033[1;33m语音已结束，Gemini 实时响应中...\033[0m]                       \n")
                    sys.stdout.flush()
                    last_cli_print_time = now

    # 建立全双工连接（常驻连接）
    handshake_done = False
    try:
        async with client.aio.live.connect(model=selected_model, config=config) as session:
            log_event("SESSION", f"Gemini Live session connected ({selected_model})")

            # 核心：如果已有上下文记忆且未由 handle 自动恢复，自动回灌前序轮次
            prefill_turns = memory.get_prefill_turns()
            if prefill_turns and not last_resumption_handle:
                try:
                    await session.send_client_content(turns=prefill_turns, turn_complete=False)
                    log_event("MEMORY_INJECTED", f"Successfully prefilled {len(prefill_turns)} history turns into session")
                    sys.stdout.write(f"\r🧠 [\033[1;36m已恢复前序 {len(prefill_turns)//2} 轮上下文对话记忆\033[0m]                       \n")
                    sys.stdout.flush()
                except Exception as e:
                    log_event("MEMORY_INJECT_FAIL", f"Failed to prefill history: {e}")
            elif last_resumption_handle:
                log_event("SESSION_RESUMED", f"Resumed session with handle {last_resumption_handle[:16]}...")
                sys.stdout.write(f"\r⚡ [\033[1;36m已通过官方 Handle 无缝恢复 Live 会话\033[0m]                       \n")
                sys.stdout.flush()

            handshake_done = True

            mic_stream = sd.RawInputStream(
                samplerate=MIC_RATE,
                channels=mic_channels,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                device=mic_idx,
                callback=mic_callback
            )
            mic_stream.start()

            # 现场底噪自适应校准 (建立近场防御网，避免安静环境下门限跌破安全值导致旁人闲聊打断)
            await asyncio.sleep(1.2)
            if not user_threshold and calib_samples:
                warm_samples = calib_samples[5:] if len(calib_samples) > 8 else calib_samples
                noise_median = int(np.median(warm_samples))
                noise_p75 = int(np.percentile(warm_samples, 75))
                noise_mean = int(np.mean(warm_samples))
                base_noise = noise_p75
                # 近场起呼门限：适配高保真近场领夹麦，保持 55~85 极速灵敏响应，绝不漏吞开头轻辅音
                START_THRESHOLD = max(55, min(85, int(base_noise * 1.25 + 12)))
                HOLD_THRESHOLD = max(30, min(45, int(START_THRESHOLD * 0.45)))
                MIN_PEAK_RMS = max(80, int(START_THRESHOLD * 1.2))

                if not user_interrupt_threshold:
                    INTERRUPT_SPEAKING_RMS = max(260, int(START_THRESHOLD * 2.8))
                    INTERRUPT_EXECUTING_RMS = max(380, int(START_THRESHOLD * 3.5))
                    INTERRUPT_THINKING_RMS = max(160, int(START_THRESHOLD * 2.0))
                else:
                    INTERRUPT_SPEAKING_RMS = user_interrupt_threshold
                    INTERRUPT_EXECUTING_RMS = max(int(user_interrupt_threshold * 1.25), 380)
                    INTERRUPT_THINKING_RMS = max(user_interrupt_threshold, 160)

                log_event("CALIBRATION", f"Noise floor median={noise_median}, mean={noise_mean}, p75={noise_p75}, start_threshold={START_THRESHOLD}, hold_threshold={HOLD_THRESHOLD}, min_peak={MIN_PEAK_RMS}, interrupt_speaking={INTERRUPT_SPEAKING_RMS}, interrupt_executing={INTERRUPT_EXECUTING_RMS}, interrupt_thinking={INTERRUPT_THINKING_RMS}")

            is_ready_to_listen = True
            sys.stdout.write(f"\r🟢 [\033[1;32m连接就绪，请直接对麦克风说话\033[0m] (起呼门限: {START_THRESHOLD}, 维持门限: {HOLD_THRESHOLD}, 打断门限: 播报{INTERRUPT_SPEAKING_RMS}/执行{INTERRUPT_EXECUTING_RMS})                 \n")
            sys.stdout.flush()

            # 看门狗：纯思考等待超时（12.0s无应答）平滑恢复就绪；失联超长超时（60s）触发重连
            async def watchdog_loop():
                while not shutdown_event.is_set() and not reconnect_event.is_set():
                    await asyncio.sleep(0.5)
                    # 1. 快速恢复：纯语音等待回复超时（12.0秒服务端无输出，自动退回 LISTENING 避免假死卡住）
                    if (
                        turn_ctrl.state == STATE_THINKING
                        and not turn_ctrl.has_active_tool
                        and not turn_ctrl.waiting_tool_summary
                        and not player.is_busy()
                    ):
                        idle_sec = time.time() - turn_ctrl.state_start_time
                        if idle_sec > 12.0:
                            log_event("THINKING_TIMEOUT_RECOVER", f"Server silent for {idle_sec:.1f}s, auto-recovering to LISTENING")
                            turn_ctrl.set_state(STATE_LISTENING)
                            sys.stdout.write(f"\r🟢 [\033[1;32m连接就绪，请直接对麦克风说话\033[0m] (起呼门限: {START_THRESHOLD})                 \n")
                            sys.stdout.flush()

                    # 1.5 孤儿执行态快速自愈：若处于 EXECUTING 状态但已无任何活动的后台工具任务，1.5 秒内自动退回 LISTENING
                    if (
                        turn_ctrl.state == STATE_EXECUTING
                        and not turn_ctrl.has_active_tool
                        and not player.is_busy()
                    ):
                        idle_sec = time.time() - turn_ctrl.state_start_time
                        if idle_sec > 1.5:
                            log_event("EXECUTING_ORPHAN_RECOVER", f"No active tool running while in EXECUTING for {idle_sec:.1f}s, auto-recovering to LISTENING")
                            turn_ctrl.set_state(STATE_LISTENING)
                            sys.stdout.write(f"\r🟢 [\033[1;32m连接就绪，请直接对麦克风说话\033[0m] (起呼门限: {START_THRESHOLD})                 \n")
                            sys.stdout.flush()

                    # 2. 彻底失联重连看门狗
                    if turn_ctrl.state in [STATE_THINKING, STATE_EXECUTING] and not player.is_busy():
                        idle_sec = time.time() - turn_ctrl.state_start_time
                        if idle_sec > 60.0:
                            log_event("WATCHDOG_TIMEOUT", f"Server unresponsive for {idle_sec:.1f}s, reconnecting")
                            sys.stdout.write("\n⚠️ [\033[1;33m云端响应超时，正在自动重连并恢复会话记忆...\033[0m]\n")
                            sys.stdout.flush()
                            reconnect_event.set()
                            break

            # 音频流发送循环 (HybridVAD 模式，带 send_lock 保护)
            async def send_loop():
                try:
                    while not shutdown_event.is_set() and not reconnect_event.is_set():
                        item = await audio_turn_queue.get()
                        if item is None:
                            break
                        if item == AudioTurnQueue.END:
                            async with send_lock:
                                await session.send_realtime_input(audio_stream_end=True)
                            log_event("SEND_AUDIO_STREAM_END", "Sent audio_stream_end=True to Gemini")
                        else:
                            async with send_lock:
                                await session.send_realtime_input(
                                    audio=types.Blob(
                                        data=item,
                                        mime_type="audio/pcm;rate=16000"
                                    )
                                )
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log_event("SEND_ERROR", str(e))
                    reconnect_event.set()

            # 异步工具执行任务：在独立后台任务中执行，完全不阻塞 recv_loop 接收打断信令
            async def execute_tools_task(target_turn_id: int, tool_call: Any, token: CancellationToken):
                current_task = asyncio.current_task()
                try:
                    if not turn_ctrl.is_current_turn(target_turn_id):
                        log_event("TOOL_CANCELLED_SKIP", f"Tools task dropped: turn {target_turn_id} is stale (current is {turn_ctrl.current_turn_id})")
                        return

                    if token.is_cancelled:
                        log_event("TOOL_TOKEN_CANCELLED", f"Tools task cancelled: reason={token.cancel_reason}, turn={target_turn_id}")
                        return
                    function_responses = []
                    for call in tool_call.function_calls:
                        if token.is_cancelled or not turn_ctrl.is_current_turn(target_turn_id):
                            break

                        func_name = call.name
                        func_args = call.args or {}

                        allowed, policy_contract = policy_manager.check_execution(
                            func_name, func_args, cancellation_token=token
                        )

                        if not allowed:
                            log_event("POLICY_BLOCK", f"Tool {func_name} blocked: {policy_contract.summary}")
                            print(f"\n🛡️  [安全策略拦截] \033[1;31m{func_name}\033[0m: {policy_contract.summary}")
                            function_responses.append(
                                types.FunctionResponse(
                                    name=func_name,
                                    id=call.id,
                                    response={"result": policy_contract.to_gemini_response()}
                                )
                            )
                            continue

                        log_event("TOOL_CALL", f"Calling {func_name} with args: {func_args}")
                        print(f"\n🛠️  [执行动作] \033[1;33m{func_name}\033[0m({func_args})")

                        t_start = time.time()
                        contract = await tool_executor.execute(func_name, func_args)

                        cost_ms = int((time.time() - t_start) * 1000)
                        log_event("TOOL_RESULT", f"{func_name} ({cost_ms}ms) status={contract.status}")
                        print(f"✨ [{func_name} 完成] ({cost_ms}ms, {contract.summary[:60]})")
                        current_tools_executed.append(f"{func_name}: {contract.summary[:60]}")

                        function_responses.append(
                            types.FunctionResponse(
                                name=func_name,
                                id=call.id,
                                response={"result": contract.to_gemini_response()}
                            )
                        )

                    if token.is_cancelled or not turn_ctrl.is_current_turn(target_turn_id):
                        log_event("TOOL_CANCELLED_DROP", f"Tools result dropped due to turn cancellation (turn {target_turn_id})")
                        return

                    if function_responses:
                        async with send_lock:
                            await session.send_tool_response(function_responses=function_responses)
                        log_event("TOOL_RESPONSE_SENT", f"Sent response for {len(function_responses)} calls, waiting for Gemini summary")
                        if turn_ctrl.is_current_turn(target_turn_id):
                            turn_ctrl.waiting_tool_summary = True
                            turn_ctrl.set_state(STATE_THINKING)
                except asyncio.CancelledError:
                    log_event("TOOL_TASK_CANCELLED", f"Active tool task cancelled for turn {target_turn_id}")
                except Exception as e:
                    log_event("TOOL_TASK_ERROR", f"Error in execute_tools_task: {e}")
                finally:
                    turn_ctrl.finish_active_tool(target_turn_id, current_task)
                    if turn_ctrl.is_current_turn(target_turn_id) and turn_ctrl.state == STATE_EXECUTING and not turn_ctrl.has_active_tool:
                        # 兜底保障：若所有工具任务均已退出且仍处于 EXECUTING 状态，自动回落至 LISTENING，杜绝假死卡住
                        log_event("TOOL_STATE_RECOVER", f"All active tool tasks ended for turn {target_turn_id}, recovering state from EXECUTING to LISTENING")
                        turn_ctrl.set_state(STATE_LISTENING)
                        sys.stdout.write(f"\r🟢 [\033[1;32m动作执行已结束，请直接对麦克风说话\033[0m]                                      \n")
                        sys.stdout.flush()

            # 接收循环：常驻监听，使用公开 session.receive() 迭代器
            async def recv_loop():
                nonlocal current_user_transcript, current_model_transcript, current_tools_executed, go_away_time_left

                try:
                    while not shutdown_event.is_set() and not reconnect_event.is_set():
                        messages_in_turn = 0

                        async for response in session.receive():
                            messages_in_turn += 1
                            if shutdown_event.is_set() or reconnect_event.is_set():
                                break

                            # 0. 检查官方会话恢复句柄更新 (Session Resumption Update)
                            if getattr(response, "session_resumption_update", None):
                                upd = response.session_resumption_update
                                if upd.resumable and upd.new_handle:
                                    session_state["handle"] = upd.new_handle
                                    log_event("RESUMPTION_HANDLE", f"Updated resumption handle: {upd.new_handle[:16]}...")

                            # 0.1 检查服务端 GoAway 通知 (平滑重连)
                            if getattr(response, "go_away", None):
                                time_left = getattr(response.go_away, "time_left", None)
                                log_event("SERVER_GO_AWAY", f"Server issued GoAway signal (time_left={time_left}), initiating instant reconnect")
                                go_away_time_left = time_left or "unspecified"
                                reconnect_event.set()
                                break

                            # 0.2 检查服务端下发的工具取消信号
                            if getattr(response, "tool_call_cancellation", None) and response.tool_call_cancellation.ids:
                                cancelled_ids = set(response.tool_call_cancellation.ids)
                                log_event("SERVER_TOOL_CANCEL", f"Server cancelled tool call IDs: {cancelled_ids}")
                                if turn_ctrl.state in [STATE_EXECUTING, STATE_THINKING]:
                                    turn_ctrl.interrupt("Server cancelled tool call")

                            # 0.3 检查与记录配额使用量统计 (Usage Metadata)
                            if getattr(response, "usage_metadata", None):
                                u = response.usage_metadata
                                log_event(
                                    "USAGE_METADATA",
                                    f"Tokens: total={u.total_token_count}, prompt={u.prompt_token_count}, response={u.response_token_count}"
                                )

                            # 1. 检查服务端打断信号 (仅在客户端正处于播报态时才打断轮次，绝不误杀客户端已开辟的新轮次Token)
                            if response.server_content and response.server_content.interrupted:
                                player.interrupt()
                                if turn_ctrl.state == STATE_SPEAKING:
                                    turn_ctrl.interrupt("Server reported interrupted")
                                log_event("SERVER_INTERRUPT", "Server reported interrupted")

                            # 1.5 语音实时转录展示与收集
                            if response.server_content:
                                if response.server_content.input_transcription and response.server_content.input_transcription.text:
                                    txt = response.server_content.input_transcription.text.strip()
                                    if txt:
                                        current_user_transcript.append(txt)
                                        log_event("USER_TRANSCRIPT", txt)
                                        sys.stdout.write(f"\n👤 [\033[1;32m用户语音转录\033[0m] {txt}\n")
                                        sys.stdout.flush()
                                if response.server_content.output_transcription and response.server_content.output_transcription.text:
                                    txt = response.server_content.output_transcription.text.strip()
                                    if txt:
                                        current_model_transcript.append(txt)
                                        log_event("MODEL_TRANSCRIPT", txt)
                                        sys.stdout.write(f"\n🤖 [\033[1;36mGemini 播报转录\033[0m] {txt}\n")
                                        sys.stdout.flush()
                                if getattr(response.server_content, "generation_complete", False):
                                    log_event("SERVER_GENERATION_COMPLETE", "Server generation completed")

                            # 2. 模型回复内容（思考、文本与音频）
                            if response.server_content and response.server_content.model_turn:
                                turn_ctrl.waiting_tool_summary = False
                                for part in response.server_content.model_turn.parts:
                                    if part.text:
                                        if getattr(part, "thought", False):
                                            log_event("MODEL_THOUGHT", part.text.strip())
                                            sys.stdout.write(f"\033[90m💭 {part.text}\033[0m")
                                        else:
                                            current_model_transcript.append(part.text)
                                            log_event("MODEL_TEXT", part.text.strip())
                                            sys.stdout.write(part.text)
                                        sys.stdout.flush()
                                    if part.inline_data:
                                        if turn_ctrl.is_interrupted or turn_ctrl.cancellation_token.is_cancelled:
                                            continue
                                        turn_ctrl.set_state(STATE_SPEAKING)
                                        player.write(part.inline_data.data)

                            # 3. 工具调用请求（异步解耦至独立后台任务，绝不阻塞 recv_loop）
                            if response.tool_call:
                                turn_ctrl.waiting_tool_summary = False
                                turn_ctrl.set_state(STATE_EXECUTING)
                                turn_id = turn_ctrl.current_turn_id
                                token = turn_ctrl.cancellation_token
                                tool_task = asyncio.create_task(
                                    execute_tools_task(turn_id, response.tool_call, token)
                                )
                                turn_ctrl.register_tool_task(tool_task)

                            # 4. 轮次终结：沉淀记忆至本地记忆池，通过 target_turn_id 隔离旧播放回调
                            # 默认模型看 turn_complete；--thinking 仅在 interaction_status=IDLE 时视为空闲
                            if turn_ctrl.is_interaction_idle(response):
                                if not response.tool_call and not turn_ctrl.has_active_tool and not turn_ctrl.waiting_tool_summary:
                                    u_txt = " ".join(current_user_transcript).strip()
                                    m_txt = "".join(current_model_transcript).strip()
                                    t_summary = "; ".join(current_tools_executed).strip()
                                    if u_txt or m_txt or t_summary:
                                        memory.record_turn(user_text=u_txt, model_text=m_txt, tool_summary=t_summary)
                                        log_event("MEMORY_RECORDED", f"Memory updated (user='{u_txt[:30]}', model='{m_txt[:30]}', tools='{t_summary[:40]}')")

                                    current_user_transcript = []
                                    current_model_transcript = []
                                    current_tools_executed = []

                                    completed_turn_id = turn_ctrl.current_turn_id

                                    async def wait_for_playback_done(target_turn_id: int):
                                        wait_start = time.time()
                                        while player.is_busy() and (time.time() - wait_start < 15.0):
                                            await asyncio.sleep(0.03)
                                        # 隔离保护：仅当仍属于当前轮次且未被打断时，才切换为 LISTENING
                                        if not turn_ctrl.is_current_turn(target_turn_id) or turn_ctrl.cancellation_token.is_cancelled:
                                            log_event("DISCARD_OLD_TURN_CALLBACK", f"Turn {target_turn_id} playback callback discarded (current turn is {turn_ctrl.current_turn_id})")
                                            return
                                        turn_ctrl.set_state(STATE_LISTENING)
                                        turn_ctrl.cancellation_token = CancellationToken(f"turn_{time.time()}")
                                        log_event("TURN_COMPLETE", f"Turn {target_turn_id} fully finished and playback done, now LISTENING")
                                        sys.stdout.write("\n🟢 [\033[1;32m就绪，请说下一句指令...\033[0m]\n")
                                        sys.stdout.flush()

                                    asyncio.create_task(wait_for_playback_done(completed_turn_id))

                        if messages_in_turn == 0 and not shutdown_event.is_set() and not reconnect_event.is_set():
                            await asyncio.sleep(0.05)

                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    log_event("RECV_ERROR", str(e))
                    reconnect_event.set()
                finally:
                    turn_ctrl.set_state(STATE_LISTENING)

            current_user_transcript = []
            current_model_transcript = []
            current_tools_executed = []
            go_away_time_left = None

            task_send = asyncio.create_task(send_loop())
            task_recv = asyncio.create_task(recv_loop())
            task_watchdog = asyncio.create_task(watchdog_loop())

            try:
                # 只有当用户退出或发生底层网络断线事件时才跳出 wait
                await asyncio.wait([task_send, task_recv, task_watchdog], return_when=asyncio.FIRST_COMPLETED)
            finally:
                task_send.cancel()
                task_recv.cancel()
                task_watchdog.cancel()
                if go_away_time_left is None:
                    turn_ctrl._cancel_tool_tasks()
                try:
                    mic_stream.stop()
                    mic_stream.close()
                except Exception:
                    pass

            if reconnect_event.is_set():
                if go_away_time_left is not None:
                    if turn_ctrl.has_active_tool:
                        try:
                            await asyncio.wait_for(
                                asyncio.shield(asyncio.gather(*list(turn_ctrl.active_tool_tasks), return_exceptions=True)),
                                timeout=1.5
                            )
                        except Exception:
                            pass
                    turn_ctrl._cancel_tool_tasks()
                    raise GoAwayReconnectError(time_left=go_away_time_left)
                raise ConnectionError("Live 连接断开或触发平滑重连 (reconnect_event)")
    except asyncio.CancelledError:
        raise
    except Exception as e:
        if last_resumption_handle and not handshake_done:
            raise ResumptionHandleExpiredError(f"Failed to resume session with handle {last_resumption_handle[:16]}...: {e}") from e
        raise


async def main():
    global NO_LOG_CONTENT

    parser = argparse.ArgumentParser(description="Gemini 3.8 Live + kimi-cu 实时语音电脑操作管家 (全双工长效记忆版)")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("GEMINI_LIVE_MODEL", MODEL_FAST),
        help=f"模型选择: {MODEL_FAST} (极速响应) 或 {MODEL_THINKING} (多步深度思考)"
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="开启深度思考模式 (使用 gemini-3.8-live-extended-thinking)"
    )
    parser.add_argument(
        "--mic",
        type=str,
        default=os.environ.get("PREFER_MIC", ""),
        help="指定优先使用的麦克风名称（留空则自动选用 macOS 系统当前默认麦克风）"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=int(os.environ.get("MIC_THRESHOLD", 0)) or None,
        help="近场说话起呼门限 RMS（默认自动根据底噪自适应，通常 140~260，防旁人闲聊误触发）"
    )
    parser.add_argument(
        "--interrupt-threshold",
        type=int,
        default=int(os.environ.get("INTERRUPT_THRESHOLD", 0)) or None,
        help="打断门限 RMS（默认根据起呼门限及状态分级自适应计算，通常 350~480）"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=os.environ.get("GEMINI_VOICE", "Aoede"),
        help="语音音色 (可选: Aoede, Puck, Charon, Kore, Fenrir)"
    )
    parser.add_argument(
        "--no-log-content",
        action="store_true",
        help="启用日志隐私脱敏，不将用户语音转写与网页正文内容持久化到磁盘"
    )
    parser.add_argument(
        "--non-strict",
        action="store_true",
        help="关闭安全策略强拦截模式（默认启用严格安全审查）"
    )
    args = parser.parse_args()

    NO_LOG_CONTENT = args.no_log_content
    selected_model = MODEL_THINKING if args.thinking else args.model
    mic_idx, mic_name, mic_channels = find_audio_devices(args.mic)
    kimi_cu_path = os.environ.get("KIMI_CU_PATH", "/Applications/KimiCU.app/Contents/MacOS/kimi-cu")

    log_event("STARTUP", f"Starting service with model={selected_model}, mic={mic_name} (ch={mic_channels}), threshold={args.threshold}, interrupt_threshold={args.interrupt_threshold}, voice={args.voice}")

    print("=" * 68)
    print("🎙️   Gemini 3.8 Live + kimi-cu 全双工长效记忆语音电脑管家")
    print(f"🤖  当前模型: \033[1;36m{selected_model}\033[0m")
    print(f"🎤  输入麦克风: \033[1;32m[{mic_idx}] {mic_name} ({mic_channels}通道)\033[0m")
    if args.threshold:
        print(f"🎯  降噪门限: \033[1;33m固定近场起呼门限 RMS={args.threshold}\033[0m (拦截远场闲聊与环境杂音)")
    else:
        print(f"🎯  降噪门限: \033[1;32m智能自适应近场门控\033[0m (开机自动采样底噪对齐，防旁人闲聊误触发)")
    if args.interrupt_threshold:
        print(f"🛑  打断门限: \033[1;33m固定打断门限 RMS={args.interrupt_threshold}\033[0m (防旁人闲聊误打断)")
    print(f"🗣️  当前音色: \033[1;35m{args.voice}\033[0m (可选: Aoede, Puck, Charon, Kore, Fenrir)")
    print(f"🧠  上下文记忆: \033[1;32m常驻长连接 + 本地记忆池自动回灌 (全流程无缝连贯)\033[0m")
    print(f"🛡️  安全策略: \033[1;32m{'严格拦截高危与注入指令' if not args.non_strict else '宽松告警模式'}\033[0m")
    print(f"📋  日志文件: \033[1;34m{LOG_FILE}\033[0m {'(已启用内容脱敏)' if NO_LOG_CONTENT else ''}")
    print("=" * 68)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n未检测到 GEMINI_API_KEY 环境变量。")
        api_key = input("👉 请输入你的 Google Gemini API Key: ").strip()
        if not api_key:
            print("❌ 必须提供 API Key 才能启动。退出。")
            return
        os.environ["GEMINI_API_KEY"] = api_key
        update_env_file(env_path, {
            "GEMINI_API_KEY": api_key,
            "GEMINI_LIVE_MODEL": selected_model,
            "PREFER_MIC": args.mic,
            "GEMINI_VOICE": args.voice,
            **({"MIC_THRESHOLD": str(args.threshold)} if args.threshold else {})
        })

    if not os.path.exists(kimi_cu_path):
        print(f"❌ 未找到 kimi-cu 可执行文件: {kimi_cu_path}")
        return

    print(f"\n[1/3] 正在启动并连接本机 kimi-cu ({kimi_cu_path})...")
    mcp_params = StdioServerParameters(
        command=kimi_cu_path,
        args=["mcp", "-s", "user"],
        env=os.environ.copy()
    )

    player = SmoothAudioPlayer(sample_rate=SPK_RATE)
    shutdown_event = asyncio.Event()
    conversation_memory = ConversationMemory(max_turns=15)

    try:
        async with stdio_client(mcp_params) as (mcp_read, mcp_write):
            async with ClientSession(mcp_read, mcp_write) as mcp_session:
                await mcp_session.initialize()
                tools_resp = await mcp_session.list_tools()

                gemini_functions = build_gemini_function_declarations(
                    tools_resp.tools,
                    is_extended_thinking="thinking" in selected_model
                )

                tool_names = [t.name for t in gemini_functions]
                print(f"✅ 成功加载 {len(gemini_functions)} 个 macOS 原生桌面与浏览器控制工具:")
                print("   " + ", ".join(tool_names))
                print(f"\n[2/3] 正在建立 Gemini Live 全双工持久长连接 ({selected_model})...")
                print(f"[3/3] 🟢 实时语音管家已就绪！")
                print(f"💡 对着 \033[1;32m{mic_name}\033[0m 说话即可全双工实时交互。按 Ctrl+C 退出。\n" + "-" * 68)

                session_state = {"handle": None}
                retry_count = 0
                while not shutdown_event.is_set():
                    try:
                        await run_session(
                            api_key=api_key,
                            selected_model=selected_model,
                            voice_name=args.voice,
                            mic_idx=mic_idx,
                            mic_name=mic_name,
                            mic_channels=mic_channels,
                            mcp_session=mcp_session,
                            gemini_functions=gemini_functions,
                            player=player,
                            shutdown_event=shutdown_event,
                            memory=conversation_memory,
                            user_threshold=args.threshold,
                            user_interrupt_threshold=args.interrupt_threshold,
                            strict_policy=not args.non_strict,
                            session_state=session_state
                        )
                        retry_count = 0
                    except asyncio.CancelledError:
                        break
                    except GoAwayReconnectError as e:
                        # 服务端计划内 GoAway 调度：立即平滑重连，不增加重试退避延迟，保留官方 Session Resumption Handle
                        log_event("GOAWAY_INSTANT_RECONNECT", f"Smoothly reconnecting following GoAway: {e}")
                        print(f"\n🔄 [服务端平滑调度 (GoAway)，立即无缝重连并恢复会话...]", file=sys.stderr)
                        retry_count = 0
                        await asyncio.sleep(0.05)
                    except ResumptionHandleExpiredError as e:
                        retry_count += 1
                        # 仅在携带 handle 建连握手失败时清空 handle 并降级为本地记忆回灌
                        log_event("RESUME_FAILED_FALLBACK", f"Session resume with handle failed: {e}. Clearing handle and falling back to memory injection.")
                        print("\n⚠️ [官方会话句柄已失效，自动清空 Handle 并降级为本地上下文记忆回灌...]", file=sys.stderr)
                        session_state["handle"] = None
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        retry_count += 1
                        # 正常网络断线或异常断开：保留 session_state["handle"] 供下一次重连尝试恢复
                        jitter = random.uniform(0.2, 0.8)
                        wait_sec = min(30.0, (1.8 ** min(retry_count, 5)) + jitter)
                        log_event("RECONNECT", f"Connection dropped (attempt {retry_count}): {e}, retrying in {wait_sec:.1f}s")
                        print(f"\n⚠️ [连接断开，{wait_sec:.1f}s 后自动重连并恢复会话 (第 {retry_count} 次)]: {e}", file=sys.stderr)
                        await asyncio.sleep(wait_sec)

    except KeyboardInterrupt:
        print("\n👋 正在退出...")
    finally:
        shutdown_event.set()
        player.stop()
        log_event("SHUTDOWN", "Service exited cleanly")
        print("\n✅ 资源已清理完毕。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
