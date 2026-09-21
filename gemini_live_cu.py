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
from typing import Optional, Any, Dict, List, Set, Union

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


class TurnController:
    """集中式轮次生命周期控制器，管理单调递增 turn_id、当前取消令牌、后台工具任务与播放结束隔离"""
    def __init__(self):
        self.current_turn_id: int = 0
        self.cancellation_token: CancellationToken = CancellationToken("turn_0")
        self.active_tool_task: Optional[asyncio.Task] = None
        self.waiting_tool_summary: bool = False
        self.has_active_tool: bool = False
        self.is_interrupted: bool = False

    def new_turn(self, reason: str = "new_turn") -> int:
        """开启新轮次：递增 turn_id，生成全新有效 CancellationToken，取消可能残余的工具任务"""
        self.current_turn_id += 1
        if self.active_tool_task and not self.active_tool_task.done():
            self.active_tool_task.cancel()
            self.active_tool_task = None
        self.cancellation_token = CancellationToken(f"turn_{self.current_turn_id}_{time.time():.3f}")
        self.has_active_tool = False
        self.waiting_tool_summary = False
        self.is_interrupted = False
        return self.current_turn_id

    def interrupt(self, reason: str = "interrupted"):
        """处理打断：取消当前 Token 与工具任务，重置状态标记"""
        self.is_interrupted = True
        self.cancellation_token.cancel(reason)
        if self.active_tool_task and not self.active_tool_task.done():
            self.active_tool_task.cancel()
            self.active_tool_task = None
        self.has_active_tool = False
        self.waiting_tool_summary = False

    def finish_active_tool(self, turn_id: int, task: Optional[asyncio.Task] = None):
        """关键修复：仅当轮次与任务对象精确匹配时才清理 has_active_tool，防止旧任务退出冲刷新任务状态"""
        if self.current_turn_id == turn_id:
            if task is None or self.active_tool_task is task:
                self.has_active_tool = False
                if self.active_tool_task is task:
                    self.active_tool_task = None

    def is_current_turn(self, turn_id: int) -> bool:
        """检查指定 turn_id 是否仍为当前最新轮次（用于丢弃迟到的旧轮次回调）"""
        return self.current_turn_id == turn_id


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
        "qq": "QQ",
        "腾讯qq": "QQ",
    }
    bid_map = {
        "calculator": "com.apple.calculator",
        "计算器": "com.apple.calculator",
        "notes": "com.apple.Notes",
        "备忘录": "com.apple.Notes",
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
2. 严禁用英文开场或打招呼！首次打招呼必须使用中文（例如：“您好！我是您的 macOS 实时语音电脑管家，请问有什么可以帮您？”）。
3. 无论用户当前使用什么语言输入、无论网页和工具返回什么语言，你的口语回复、思考总结、状态反馈必须全部使用纯正自然的中文！

【核心身份与上下文记忆原则】：
1. 你具备跨轮次的上下文记忆能力！你能够清晰记住用户在前面各轮说过的指令、参数、偏好以及上一步打开的应用与网页。
2. 当用户提及“刚才”、“之前”、“上一个”等指代词时，你必须根据前序对话历史做出准确衔接与回应！

你是一个运行在 macOS 上的实时语音电脑操作管家。
你能操控 macOS 桌面上的所有应用和窗口，并具备强劲的网页浏览与检索能力。

【工具能力与使用规则】：
1. 应用程序管理（macOS 本地）：
   - open_app: 打开或前台激活任意应用程序（计算器、备忘录、微信、QQ、Outlook、Word、音乐、Pages 等）。用户要求“打开XXX软件”时必须优先调用此工具！
   - list_apps: 查看当前正在运行的所有应用程序（包含 name, bundle_id, pid）。操作前可用此工具确认目标应用是否在运行。
   - get_app_state: 获取指定桌面应用的窗口控件树（参数 app 传 bundle_id，mode 建议使用 'ax' 获取极速控件树）。
   - click: 点击目标桌面应用的控件（传 app 和控件 index，或点击坐标 x, y）。
   - type_text: 向目标应用输入文本（支持 clear=True 清空后输入，submit=True 按回车提交）。
   - press_key: 向应用发送按键或快捷键（如 return, enter, escape, cmd+c, cmd+v, cmd+w, cmd+n 等，参数 app 传 bundle_id）。
   - scroll: 滚动桌面应用窗口。

2. 网页浏览与联网搜索（基于 Ego Lite 极速浏览器）：
   - browser_open: 打开指定网址或 URL，返回网页标题与精简正文（如 browser_open(url="https://www.ithome.com")）。
   - browser_search: 在浏览器中搜索关键词并提炼要点（如 browser_search(query="特斯拉 Roadster 最新售价")）。
   - browser_get_content: 抓取当前已打开网页的正文内容并总结。
   - browser_list_actions: 列出当前网页中所有可交互操作元素（链接、按钮）及稳定编号 [#ID]。当页面选项较多或可能出现重复文案误点击时，先调用此工具列出候选编号！
   - browser_click: 在当前网页中点击指定链接或按钮（支持传入候选编号如 "#1" 或标题文字）。
   - browser_scroll: 在当前网页中向上或向下滚动（如 browser_scroll(direction="down")）。
   【重要原则】：所有网页访问、互联网资讯搜索、网页内容阅读一律优先使用 browser_* 系列工具！当用户要求“看网页/打开某网站”时，直接使用 browser_open，绝不需要多此一举去调用 open_app("Google Chrome")。

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

# 状态枚举
STATE_LISTENING = "LISTENING"  # 空闲听用户说话
STATE_THINKING = "THINKING"    # 用户已说完，正在等待 Gemini 思考/下发动作
STATE_EXECUTING = "EXECUTING"  # kimi-cu 工具正在物理执行中
STATE_SPEAKING = "SPEAKING"    # 扬声器正在平滑播报中


def find_audio_devices(preferred_mic="Wireless Mic Rx"):
    """智能查找麦克风，自动检测支持的通道数"""
    devices = sd.query_devices()
    mic_idx = None
    mic_name = "系统默认麦克风"
    channels = 1

    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0:
            if preferred_mic.lower() in d.get("name", "").lower():
                mic_idx = idx
                mic_name = d["name"]
                channels = min(2, d.get("max_input_channels", 1))
                break

    if mic_idx is None:
        default_in = sd.default.device[0]
        if default_in is not None and default_in >= 0:
            mic_idx = default_in
            mic_name = devices[default_in]["name"]
            channels = min(2, devices[default_in].get("max_input_channels", 1))
        else:
            mic_idx = 0
            mic_name = devices[0]["name"]
            channels = min(2, devices[0].get("max_input_channels", 1))

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

    if session_state is None:
        session_state = {"handle": None}

    thinking_config = None
    if "thinking" in selected_model:
        thinking_config = types.ThinkingConfig(include_thoughts=True)

    last_resumption_handle = session_state.get("handle")
    session_resumption_cfg = (
        types.SessionResumptionConfig(handle=last_resumption_handle)
        if last_resumption_handle
        else types.SessionResumptionConfig()
    )
    context_compression_cfg = types.ContextWindowCompressionConfig(
        sliding_window=types.SlidingWindow(target_tokens=4000)
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
        input_audio_transcription=types.AudioTranscriptionConfig(),
        output_audio_transcription=types.AudioTranscriptionConfig(),
        thinking_config=thinking_config,
        tools=[types.Tool(function_declarations=gemini_functions)],
        session_resumption=session_resumption_cfg,
        context_window_compression=context_compression_cfg,
    )

    state = STATE_LISTENING
    state_start_time = time.time()
    loop = asyncio.get_running_loop()

    audio_stream_queue = asyncio.Queue(maxsize=250)
    AUDIO_STREAM_END = b"__AUDIO_STREAM_END__"
    reconnect_event = asyncio.Event()

    # 近场降噪门限参数 (初始基准值，稍后自适应对齐)
    START_THRESHOLD = user_threshold or 110
    HOLD_THRESHOLD = 50
    ATTACK_FRAMES = 2
    SILENCE_CHUNKS = 10
    INTERRUPT_RMS = 220
    MIN_PEAK_RMS = 95

    pre_roll = collections.deque(maxlen=4)
    is_speaking = False
    attack_count = 0
    silence_count = 0
    interrupt_frames = 0
    is_ready_to_listen = False
    calib_samples = []
    last_cli_print_time = 0.0

    turn_ctrl = TurnController()
    send_lock = asyncio.Lock()

    def set_state(new_state):
        nonlocal state, state_start_time, interrupt_frames
        if state != new_state:
            state = new_state
            state_start_time = time.time()
            interrupt_frames = 0
            log_event("STATE", f"Transitioned to {new_state}")

    def safe_put_audio(chunk_bytes):
        try:
            audio_stream_queue.put_nowait(chunk_bytes)
        except asyncio.QueueFull:
            try:
                audio_stream_queue.get_nowait()
            except Exception:
                pass
            try:
                audio_stream_queue.put_nowait(chunk_bytes)
            except Exception:
                pass

    def drain_audio_queue():
        while not audio_stream_queue.empty():
            try:
                audio_stream_queue.get_nowait()
            except Exception:
                break

    def handle_voice_barge_in(reason: str, initial_chunk: Optional[bytes] = None):
        """在事件循环主线程中线程安全地处理打断、清理队列与开启新轮次"""
        turn_ctrl.interrupt(reason)
        drain_audio_queue()
        turn_ctrl.new_turn(f"speech_after_{reason}")
        if initial_chunk:
            safe_put_audio(initial_chunk)

    def handle_voice_speech_start(chunks_to_send: list):
        """在事件循环主线程中线程安全地开启新轮次并推送起呼前摇音频"""
        turn_ctrl.new_turn("speech_start")
        for pr_chunk in chunks_to_send:
            safe_put_audio(pr_chunk)

    def mic_callback(indata, frames, time_info, status):
        nonlocal is_speaking, attack_count, silence_count, state, interrupt_frames
        nonlocal START_THRESHOLD, HOLD_THRESHOLD, MIN_PEAK_RMS, INTERRUPT_RMS
        nonlocal last_cli_print_time

        if mic_channels == 2:
            stereo = np.frombuffer(indata, dtype=np.int16).reshape(-1, 2)
            ch0 = stereo[:, 0]
            ch1 = stereo[:, 1]
            rms0 = int(np.sqrt(np.mean(ch0.astype(np.float32)**2)))
            rms1 = int(np.sqrt(np.mean(ch1.astype(np.float32)**2)))
            if rms0 > rms1 * 1.8 and rms0 > 25:
                mono_samples = ch0
                rms = rms0
            elif rms1 > rms0 * 1.8 and rms1 > 25:
                mono_samples = ch1
                rms = rms1
            else:
                mono_samples = ((ch0.astype(np.int32) + ch1.astype(np.int32)) // 2).astype(np.int16)
                rms = max(rms0, rms1)
            raw_bytes = mono_samples.tobytes()
        else:
            mono_samples = np.frombuffer(indata, dtype=np.int16)
            rms = int(np.sqrt(np.mean(mono_samples.astype(np.float32)**2)))
            raw_bytes = bytes(indata)

        if not is_ready_to_listen:
            calib_samples.append(rms)
            return

        now = time.time()
        should_print_cli = (now - last_cli_print_time >= 0.1)
        bars = "▇" * min(12, rms // 30) + "░" * max(0, 12 - rms // 30)

        # 1. 扬声器播报中：检测打断
        if state == STATE_SPEAKING:
            if rms >= INTERRUPT_RMS:
                interrupt_frames += 1
                if interrupt_frames >= 2:
                    player.interrupt()
                    loop.call_soon_threadsafe(handle_voice_barge_in, "speaking", raw_bytes)
                    set_state(STATE_LISTENING)
                    is_speaking = True
                    attack_count = 0
                    silence_count = 0
                    log_event("USER_INTERRUPT", f"User interrupted speaking (rms={rms})")
                    sys.stdout.write(f"\r🛑 [\033[1;31m已打断播报，请继续说...\033[0m]                           \n")
                    sys.stdout.flush()
                    last_cli_print_time = now
            else:
                interrupt_frames = 0
            if should_print_cli:
                sys.stdout.write(f"\r🗣️ [\033[1;35mGemini 正在播报...\033[0m] 音量: |{bars}| ({rms:3d}) ")
                sys.stdout.flush()
                last_cli_print_time = now
            return

        # 2. kimi-cu 正在执行动作：检测到强力打断停止后续动作
        if state == STATE_EXECUTING:
            if rms >= INTERRUPT_RMS:
                interrupt_frames += 1
                if interrupt_frames >= 3:
                    player.interrupt()
                    loop.call_soon_threadsafe(handle_voice_barge_in, "executing", raw_bytes)
                    set_state(STATE_LISTENING)
                    is_speaking = True
                    attack_count = 0
                    silence_count = 0
                    log_event("USER_INTERRUPT", f"User interrupted executing state (rms={rms})")
                    sys.stdout.write(f"\r🛑 [\033[1;31m已打断动作执行，请继续说...\033[0m]                           \n")
                    sys.stdout.flush()
                    last_cli_print_time = now
            else:
                interrupt_frames = 0
            if should_print_cli:
                sys.stdout.write(f"\r⚙️ [\033[1;33mkimi-cu 正在执行桌面动作...\033[0m] 音量: |{bars}| ({rms:3d}) ")
                sys.stdout.flush()
                last_cli_print_time = now
            return

        # 3. 正在思考中
        if state == STATE_THINKING:
            if rms >= max(260, int(START_THRESHOLD * 1.8)):
                interrupt_frames += 1
                if interrupt_frames >= 3:
                    player.interrupt()
                    loop.call_soon_threadsafe(handle_voice_barge_in, "thinking", raw_bytes)
                    set_state(STATE_LISTENING)
                    is_speaking = True
                    attack_count = 0
                    silence_count = 0
                    log_event("USER_INTERRUPT", f"User interrupted thinking state (rms={rms})")
                    sys.stdout.write(f"\r🛑 [\033[1;31m已取消等待，请重新说...\033[0m]                           \n")
                    sys.stdout.flush()
                    last_cli_print_time = now
            else:
                interrupt_frames = 0
            if should_print_cli:
                sys.stdout.write(f"\r🧠 [\033[1;36mGemini 正在处理中...\033[0m] 音量: |{bars}| ({rms:3d}) ")
                sys.stdout.flush()
                last_cli_print_time = now
            return

        # 4. 空闲监听中 (HybridVAD 连续流式输入)
        if not is_speaking:
            if rms >= START_THRESHOLD:
                attack_count += 1
                pre_roll.append(raw_bytes)
                if should_print_cli or attack_count == 1:
                    sys.stdout.write(f"\r🎤 [\033[1;33m检测到声音 {attack_count}/{ATTACK_FRAMES}\033[0m] 音量: |{bars}| ({rms:3d}/{START_THRESHOLD}) ")
                    sys.stdout.flush()
                    last_cli_print_time = now

                if attack_count >= ATTACK_FRAMES:
                    is_speaking = True
                    silence_count = 0
                    attack_count = 0
                    chunks_to_send = list(pre_roll)
                    pre_roll.clear()
                    loop.call_soon_threadsafe(handle_voice_speech_start, chunks_to_send)
                    log_event("USER_SPEECH_START", f"Streaming speech started (rms={rms})")
            else:
                attack_count = 0
                pre_roll.append(raw_bytes)
                if should_print_cli:
                    if rms >= max(35, int(START_THRESHOLD * 0.55)):
                        sys.stdout.write(f"\r🎤 [\033[93m收音中\033[0m] 音量: |{bars}| ({rms:3d}/{START_THRESHOLD}) ")
                    else:
                        sys.stdout.write(f"\r🎤 [\033[90m监听中\033[0m] 音量: |{bars}| ({rms:3d}/{START_THRESHOLD}) ")
                    sys.stdout.flush()
                    last_cli_print_time = now
        else:
            # 持续流式推送 PCM 块
            loop.call_soon_threadsafe(safe_put_audio, raw_bytes)
            if rms >= HOLD_THRESHOLD:
                silence_count = 0
                if should_print_cli:
                    sys.stdout.write(f"\r🎤 [\033[1;32m正在流式说话\033[0m] 音量: |{bars}| ({rms:3d}) ")
                    sys.stdout.flush()
                    last_cli_print_time = now
            else:
                silence_count += 1
                if should_print_cli:
                    sys.stdout.write(f"\r🎤 [\033[1;36m检测停顿 {silence_count}/{SILENCE_CHUNKS}\033[0m] 音量: |{bars}| ({rms:3d}) ")
                    sys.stdout.flush()
                    last_cli_print_time = now

                if silence_count >= SILENCE_CHUNKS:
                    is_speaking = False
                    silence_count = 0
                    attack_count = 0
                    # 发送音频流结束标记，通知 Gemini 即刻开始响应
                    loop.call_soon_threadsafe(safe_put_audio, AUDIO_STREAM_END)
                    set_state(STATE_THINKING)
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

        # 现场底噪自适应校准
        await asyncio.sleep(1.2)
        if not user_threshold and calib_samples:
            warm_samples = calib_samples[5:] if len(calib_samples) > 8 else calib_samples
            noise_median = int(np.median(warm_samples))
            noise_p75 = int(np.percentile(warm_samples, 75))
            noise_mean = int(np.mean(warm_samples))
            base_noise = noise_p75
            START_THRESHOLD = max(65, min(160, int(base_noise * 1.7 + 25)))
            HOLD_THRESHOLD = max(35, min(90, int(base_noise * 1.1 + 10)))
            MIN_PEAK_RMS = max(75, int(START_THRESHOLD * 1.15))
            INTERRUPT_RMS = max(160, int(START_THRESHOLD * 1.7))
            log_event("CALIBRATION", f"Noise floor median={noise_median}, mean={noise_mean}, p75={noise_p75}, start_threshold={START_THRESHOLD}, hold_threshold={HOLD_THRESHOLD}, min_peak={MIN_PEAK_RMS}")

        is_ready_to_listen = True
        sys.stdout.write(f"\r🟢 [\033[1;32m连接就绪，请直接对麦克风说话\033[0m] (起呼门限: {START_THRESHOLD}, 维持门限: {HOLD_THRESHOLD})                 \n")
        sys.stdout.flush()

        # 看门狗：仅在真正彻底失联时触发重连，不误杀正常动作
        async def watchdog_loop():
            while not shutdown_event.is_set() and not reconnect_event.is_set():
                await asyncio.sleep(1.0)
                if state in [STATE_THINKING, STATE_EXECUTING] and not player.is_busy():
                    idle_sec = time.time() - state_start_time
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
                    item = await audio_stream_queue.get()
                    if item is None:
                        break
                    if item == AUDIO_STREAM_END:
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
                if token.is_cancelled or not turn_ctrl.is_current_turn(target_turn_id):
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

                    if func_name in ["press_key", "type_text", "click"]:
                        if not func_args.get("app") and not func_args.get("pid"):
                            func_args["app"] = "com.apple.finder"

                    if func_name in ["press_key", "type_text"]:
                        func_args.setdefault("activate", True)
                    elif func_name == "get_app_state":
                        func_args.setdefault("mode", "ax")

                    t_start = time.time()
                    try:
                        if func_name == "open_app":
                            target_app = func_args.get("name", "") or func_args.get("app", "")
                            target_bid = func_args.get("bundle_id", "")
                            raw_text = await asyncio.to_thread(launch_mac_app, target_app, target_bid)
                            contract = ToolResultContract(
                                ok="成功" in raw_text,
                                action="open_app",
                                status="success" if "成功" in raw_text else "error",
                                summary=raw_text,
                                side_effects="app_launched"
                            )
                        elif func_name == "browser_open":
                            url = func_args.get("url", "") or func_args.get("url_or_kw", "")
                            res_text = await browser_open(url)
                            err = is_browser_error(res_text)
                            contract = ToolResultContract(
                                ok=not err,
                                action="browser_open",
                                status="success" if not err else "error",
                                summary="已打开网页" if not err else res_text[:60],
                                data=res_text,
                                side_effects="navigation" if not err else "none",
                                error=res_text if err else None
                            )
                        elif func_name == "browser_search":
                            query = func_args.get("query", "")
                            engine = func_args.get("engine", "baidu")
                            res_text = await browser_search(query, engine)
                            err = is_browser_error(res_text)
                            contract = ToolResultContract(
                                ok=not err,
                                action="browser_search",
                                status="success" if not err else "error",
                                summary=f"已搜索关键词 '{query}'" if not err else res_text[:60],
                                data=res_text,
                                side_effects="navigation" if not err else "none",
                                error=res_text if err else None
                            )
                        elif func_name == "browser_get_content":
                            res_text = await browser_get_content()
                            err = is_browser_error(res_text)
                            contract = ToolResultContract(
                                ok=not err,
                                action="browser_get_content",
                                status="success" if not err else "error",
                                summary="已提取页面内容" if not err else res_text[:60],
                                data=res_text,
                                error=res_text if err else None
                            )
                        elif func_name == "browser_list_actions":
                            max_items = func_args.get("max_items", 25)
                            res_text = await browser_list_actions(max_items=max_items)
                            err = is_browser_error(res_text)
                            contract = ToolResultContract(
                                ok=not err,
                                action="browser_list_actions",
                                status="success" if not err else "error",
                                summary="已获取页面可操作候选项" if not err else res_text[:60],
                                data=res_text,
                                error=res_text if err else None
                            )
                        elif func_name == "browser_click":
                            target = func_args.get("text", "") or func_args.get("text_or_selector", "") or func_args.get("target", "")
                            res_text = await browser_click(target)
                            err = is_browser_error(res_text) or ("已成功点击" not in res_text)
                            contract = ToolResultContract(
                                ok=not err,
                                action="browser_click",
                                status="success" if not err else "error",
                                summary=res_text[:80],
                                side_effects="ui_updated" if not err else "none",
                                error=res_text if err else None
                            )
                        elif func_name == "browser_scroll":
                            direction = func_args.get("direction", "down")
                            res_text = await browser_scroll(direction)
                            err = is_browser_error(res_text)
                            contract = ToolResultContract(
                                ok=not err,
                                action="browser_scroll",
                                status="success" if not err else "error",
                                summary=res_text[:80],
                                side_effects="ui_updated" if not err else "none",
                                error=res_text if err else None
                            )
                        else:
                            mcp_res = await mcp_session.call_tool(func_name, func_args)
                            is_err = bool(getattr(mcp_res, "is_error", False) or getattr(mcp_res, "isError", False))
                            texts = []
                            for item in mcp_res.content:
                                if hasattr(item, "text") and item.text:
                                    texts.append(item.text)
                                elif hasattr(item, "data"):
                                    texts.append("[截图像素数据已捕获]")
                            raw_text = "\n".join(texts) if texts else ("MCP 执行失败" if is_err else "ok")
                            cleaned_text = format_tool_result(func_name, raw_text)
                            contract = ToolResultContract(
                                ok=not is_err,
                                action=func_name,
                                status="success" if not is_err else "error",
                                summary=f"已执行 {func_name}" if not is_err else f"{func_name} 执行失败",
                                data=cleaned_text,
                                side_effects="ui_updated" if not is_err else "none",
                                error=cleaned_text if is_err else None
                            )
                    except asyncio.CancelledError:
                        raise
                    except Exception as err:
                        contract = ToolResultContract(
                            ok=False,
                            action=func_name,
                            status="error",
                            summary="执行异常",
                            error=str(err)
                        )

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
                        set_state(STATE_THINKING)
            except asyncio.CancelledError:
                log_event("TOOL_TASK_CANCELLED", f"Active tool task cancelled for turn {target_turn_id}")
            except Exception as e:
                log_event("TOOL_TASK_ERROR", f"Error in execute_tools_task: {e}")
            finally:
                turn_ctrl.finish_active_tool(target_turn_id, current_task)

        # 接收循环：常驻监听，使用公开 session.receive() 迭代器
        async def recv_loop():
            nonlocal state
            nonlocal current_user_transcript, current_model_transcript, current_tools_executed

            try:
                while not shutdown_event.is_set() and not reconnect_event.is_set():
                    policy_manager.reset_turn()
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
                            log_event("SERVER_GO_AWAY", "Server issued GoAway signal, graceful reconnect triggered")
                            reconnect_event.set()
                            break

                        # 0.2 检查服务端下发的工具取消信号
                        if getattr(response, "tool_call_cancellation", None) and response.tool_call_cancellation.ids:
                            cancelled_ids = set(response.tool_call_cancellation.ids)
                            log_event("SERVER_TOOL_CANCEL", f"Server cancelled tool call IDs: {cancelled_ids}")
                            turn_ctrl.interrupt("Server cancelled tool call")

                        # 1. 检查服务端打断信号
                        if response.server_content and response.server_content.interrupted:
                            player.interrupt()
                            turn_ctrl.interrupt("Server reported interrupted")
                            set_state(STATE_LISTENING)
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
                                    set_state(STATE_SPEAKING)
                                    player.write(part.inline_data.data)

                        # 3. 工具调用请求（异步解耦至独立后台任务，绝不阻塞 recv_loop）
                        if response.tool_call:
                            turn_ctrl.has_active_tool = True
                            turn_ctrl.waiting_tool_summary = False
                            set_state(STATE_EXECUTING)
                            turn_id = turn_ctrl.current_turn_id
                            token = turn_ctrl.cancellation_token
                            turn_ctrl.active_tool_task = asyncio.create_task(
                                execute_tools_task(turn_id, response.tool_call, token)
                            )

                        # 4. 轮次终结：沉淀记忆至本地记忆池，通过 target_turn_id 隔离旧播放回调
                        if response.server_content and response.server_content.turn_complete:
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
                                        await asyncio.sleep(0.05)
                                    await asyncio.sleep(0.1)
                                    # 隔离保护：仅当仍属于当前轮次且未被打断时，才切换为 LISTENING
                                    if not turn_ctrl.is_current_turn(target_turn_id) or turn_ctrl.cancellation_token.is_cancelled:
                                        log_event("DISCARD_OLD_TURN_CALLBACK", f"Turn {target_turn_id} playback callback discarded (current turn is {turn_ctrl.current_turn_id})")
                                        return
                                    set_state(STATE_LISTENING)
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
                set_state(STATE_LISTENING)

        current_user_transcript = []
        current_model_transcript = []
        current_tools_executed = []

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
            if turn_ctrl.active_tool_task and not turn_ctrl.active_tool_task.done():
                turn_ctrl.active_tool_task.cancel()
            try:
                mic_stream.stop()
                mic_stream.close()
            except Exception:
                pass

        if reconnect_event.is_set():
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
        default=os.environ.get("PREFER_MIC", "Wireless Mic Rx"),
        help="指定优先使用的麦克风名称（默认: Wireless Mic Rx）"
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=int(os.environ.get("MIC_THRESHOLD", 0)) or None,
        help="近场说话起呼门限 RMS（默认自动根据底噪自适应，通常 80~150）"
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

    log_event("STARTUP", f"Starting service with model={selected_model}, mic={mic_name} (ch={mic_channels}), threshold={args.threshold}, voice={args.voice}")

    print("=" * 68)
    print("🎙️   Gemini 3.8 Live + kimi-cu 全双工长效记忆语音电脑管家")
    print(f"🤖  当前模型: \033[1;36m{selected_model}\033[0m")
    print(f"🎤  输入麦克风: \033[1;32m[{mic_idx}] {mic_name} ({mic_channels}通道)\033[0m")
    if args.threshold:
        print(f"🎯  降噪门限: \033[1;33m固定近场门限 RMS={args.threshold}\033[0m (拦截远场闲聊与环境杂音)")
    else:
        print(f"🎯  降噪门限: \033[1;32m智能自适应近场门控\033[0m (开机自动采样底噪对齐)")
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

                gemini_functions = [
                    types.FunctionDeclaration(
                        name=t.name,
                        description=t.description or "",
                        parameters=t.input_schema or {"type": "object", "properties": {}}
                    )
                    for t in tools_resp.tools
                ]

                # 注入 macOS 原生应用启动工具
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
                    }
                )
                gemini_functions.append(open_app_tool)
                gemini_functions.extend(get_browser_function_declarations())

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
                            strict_policy=not args.non_strict,
                            session_state=session_state
                        )
                        retry_count = 0
                    except asyncio.CancelledError:
                        break
                    except ResumptionHandleExpiredError as e:
                        retry_count += 1
                        # 仅在携带 handle 建连握手失败时清空 handle 并降级为本地记忆回灌
                        log_event("RESUME_FAILED_FALLBACK", f"Session resume with handle failed: {e}. Clearing handle and falling back to memory injection.")
                        print("\n⚠️ [官方会话句柄已失效，自动清空 Handle 并降级为本地上下文记忆回灌...]", file=sys.stderr)
                        session_state["handle"] = None
                        await asyncio.sleep(0.5)
                    except Exception as e:
                        retry_count += 1
                        # 正常网络断线或服务端 GoAway：保留 session_state["handle"] 供下一次重连尝试恢复
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
