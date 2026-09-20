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
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import numpy as np
import sounddevice as sd

# 加载配置
CUR_DIR = Path(__file__).parent
env_path = CUR_DIR / ".env"
load_dotenv(dotenv_path=env_path)
load_dotenv()

# 日志持久化文件
LOG_FILE = CUR_DIR / "gemini_live_cu.log"

def log_event(category: str, message: str):
    """写入结构化历史记录文件供排查与对比"""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{category}] {message}\n"
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

def clean_ax_text(raw_text: str, max_chars: int = 1800) -> str:
    """智能清洗 AX 树：提取真正的页面文本、标题和链接，滤除无用布局容器与系统菜单噪声"""
    header_lines = []
    meaningful_lines = []
    ignore_noise = {
        "后退", "前进", "共享", "添加到阅读列表", "显示侧边栏", "隐藏侧边栏",
        "标签页概览", "新标签页", "下载项", "起始页", "关闭", "取消", "菜单栏",
        "Apple", "Safari", "File", "Edit", "View", "History", "Bookmarks", "Develop", "Window", "Help"
    }

    for line in raw_text.splitlines():
        s = line.strip()
        if s.startswith("#") or s.startswith("- app:") or s.startswith("- window_title:"):
            header_lines.append(s)
            continue
        if any(k in s for k in ["AXStaticText", "AXHeading", "AXLink", "AXTitle", "AXValue"]):
            quotes = re.findall(r'\"([^\"]+)\"', s)
            for q in quotes:
                q = q.strip()
                if len(q) >= 2 and q not in ignore_noise and not q.startswith("http"):
                    meaningful_lines.append(q)
            if not quotes and "=" in s:
                val = s.split("=")[-1].strip()
                if len(val) >= 2 and val not in ignore_noise and not val.startswith("http"):
                    meaningful_lines.append(val)

    header = "\n".join(header_lines)
    unique_lines = list(dict.fromkeys(meaningful_lines))
    
    if not unique_lines:
        return (
            header + "\n\n【提示】：当前应用暂无打开的前台主窗口或未加载出页面正文（仅检测到系统菜单栏）。"
            "若需打开页面，可调用 press_key 打开新窗口(cmd+n)或新标签(cmd+t)。"
        )

    body = "\n".join(unique_lines)
    result = header + "\n\n【提取的页面正文与新闻要点】：\n" + body
    if len(result) > max_chars:
        result = result[:max_chars] + "\n...(已精简提取)"
    return result

# 自动配置本地代理端口（若未显式配置）
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
你是一个运行在 macOS 上的实时语音电脑操作管家。
你的所有电脑操作都完全通过本机的 kimi-cu MCP 工具执行（严禁调用系统外挂或假命令）。

你拥有的 kimi-cu MCP 原生工具包含：
- list_apps: 查看当前正在运行的所有应用程序（包含 name, bundle_id, pid）。操作前优先用此工具确认目标应用是否在运行。
- get_app_state: 获取指定应用的窗口控件树（参数 app 传 bundle_id，mode 建议使用 'ax' 获取极速控件树，每个控件都有唯一的 index）。
- click: 点击目标应用的控件（传 app 和控件 index，或点击坐标 x, y）。
- type_text: 向目标应用输入文本（支持 clear=True 清空后输入，submit=True 按回车提交）。
- press_key: 向应用发送按键或快捷键（如 return, enter, escape, cmd+space, cmd+c, cmd+v, cmd+w, ctrl+cmd+f 等，参数 app 传 bundle_id）。
- scroll: 滚动应用窗口。
- set_value: 直接设置输入框内容。
- select_text: 选取文本内容。
- drag / drag_paths: 拖拽操作。

【浏览器（Safari / Chrome）识别与操作规范】：
1. 查找浏览器：当用户提到“浏览器”时，先调用 list_apps 检查正在运行的浏览器：
   - Safari: bundle_id 为 "com.apple.Safari"
   - Google Chrome: bundle_id 为 "com.google.Chrome"
2. 将浏览器放到当前屏幕 / 激活置顶全屏展示：
   - 当用户要求“把浏览器放到当前屏幕”、“打开浏览器”、“最大化浏览器”时，调用 press_key(app="com.apple.Safari", keys="ctrl+cmd+f", activate=True) 直接将 Safari 窗口在当前主屏幕置顶全屏展示！
3. 总结或查看网页内容：
   - 直接调用 get_app_state(app="com.apple.Safari", mode="ax") 获取网页文本。拿到返回的页面主要文本与新闻后，用自然流畅的中文口语直接向用户总结重点（提炼 2-3 条核心标题或要点）。
   - 若系统提示“当前应用暂无打开的前台主窗口”，可先调用 press_key(app="com.apple.Safari", keys="cmd+n", activate=True) 打开新窗口。
4. 浏览器内常用操作：
   - 聚焦地址栏并输入网址：先调用 press_key(app="com.apple.Safari", keys="cmd+l", activate=True)，随后调用 type_text(app="com.apple.Safari", text="目标网址", clear=True, submit=True)。
   - 刷新页面：press_key(app="com.apple.Safari", keys="cmd+r")
   - 新建标签：press_key(app="com.apple.Safari", keys="cmd+t")
   - 关闭标签：press_key(app="com.apple.Safari", keys="cmd+w")
   - 滚动页面：scroll(app="com.apple.Safari", count=5, direction="down")

【核心交互与口语原则】：
1. 【静默执行，最终统一汇报（Action-First, Silent Execution）】：
   - 当需要调用工具操作电脑时，直接调用 kimi-cu 工具执行！
   - 严禁在调用工具前说废话（例如“好的，正在为您打开”、“我来帮您看”）；
   - 严禁在多步工具执行的中间过程口头提示“第X步已完成”、“操作已完成”、“已打开浏览器”等；
   - 保持安静迅速地执行所有动作。只有当全部动作彻底完成，或者拿到网页内容后，才一次性用简练自然的中文口语向用户做最终总结汇报！
2. 【日常问答自然连贯】：
   - 如果用户只是打招呼或咨询日常问题（不涉及电脑操作），一次性把话说完整，切忌断词卡壳。
"""

# 音频参数
MIC_RATE = 16000     # 录音采样率 (16kHz, int16 单声道)
SPK_RATE = 24000     # 播音采样率 (24kHz, int16 单声道)
CHUNK_SIZE = 1024    # 64ms 块

# 状态枚举
STATE_LISTENING = "LISTENING"  # 空闲听用户说话（麦克风随时接话）
STATE_THINKING = "THINKING"    # 用户已说完，正在等待 Gemini 思考/下发首个动作
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
    def __init__(self, sample_rate=SPK_RATE):
        self.sample_rate = sample_rate
        self.queue = queue.Queue()
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
                        # 1. 空闲启动时，先做微小预缓冲（积攒 2-3 块或最多等 100ms），吸收网络抖动
                        data = self.queue.get(timeout=0.05)
                        if data is None:
                            break
                        
                        self.is_playing = True
                        stream.write(data)

                        # 2. 连续播放循环
                        while self.running:
                            try:
                                chunk = self.queue.get(timeout=0.15)
                                if chunk is None:
                                    return
                                stream.write(chunk)
                            except queue.Empty:
                                # 超过 150ms 仍无后续音频，判定本轮语音播放结束
                                self.is_playing = False
                                break
                    except queue.Empty:
                        self.is_playing = False
                        continue
        except Exception as e:
            log_event("AUDIO_ERROR", str(e))

    def write(self, data: bytes):
        if self.running and data:
            self.queue.put(data)

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
        self.queue.put(None)
        self.thread.join(timeout=0.5)


async def run_session(api_key, selected_model, voice_name, mic_idx, mic_name, mic_channels, mcp_session, gemini_functions, player, shutdown_event, user_threshold=None):
    """单个全双工会话生命周期（智能双门限近场语音门控 + 状态严密闭环 + 看门狗超时自愈）"""
    client = genai.Client(api_key=api_key)

    thinking_config = None
    if "thinking" in selected_model:
        thinking_config = types.ThinkingConfig(include_thoughts=True)

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
        thinking_config=thinking_config,
        tools=[types.Tool(function_declarations=gemini_functions)]
    )

    state = STATE_LISTENING
    state_start_time = time.time()
    loop = asyncio.get_running_loop()
    turn_queue = asyncio.Queue()
    reconnect_event = asyncio.Event()

    # 近场降噪门限参数 (初始安全基准值，稍后由现场底噪校准自动自适应)
    START_THRESHOLD = user_threshold or 110  # 佩戴者起呼门限（默认110，轻声也能灵敏唤醒）
    HOLD_THRESHOLD = 50                      # 语音维持门限（保护句尾弱音）
    ATTACK_FRAMES = 2                        # 连续 2 帧（~128ms）超门限即确认（过滤瞬时单帧尖峰，人声即刻唤醒）
    SILENCE_CHUNKS = 13                      # 停顿检测帧数 (~0.85秒判定说话结束)
    INTERRUPT_RMS = 240                      # 强行打断门限
    MIN_PEAK_RMS = 95                        # 说话整句必须达到的峰值（拦截微弱环境底噪漂移）
    MIN_DURATION = 0.35                      # 最短有效说话时长（支持简短指令）
    MIN_VOICED_RATIO = 0.05                  # 有效高能帧比例 (5%即可，支持轻音指令)

    audio_buffer = []
    rms_history = []
    pre_roll = collections.deque(maxlen=4)
    is_speaking = False
    attack_count = 0
    silence_count = 0
    interrupt_frames = 0
    is_ready_to_listen = False
    has_active_tool = False
    waiting_tool_summary = False
    calib_samples = []

    def set_state(new_state):
        nonlocal state, state_start_time, interrupt_frames
        if state != new_state:
            state = new_state
            state_start_time = time.time()
            interrupt_frames = 0
            log_event("STATE", f"Transitioned to {new_state}")

    def mic_callback(indata, frames, time_info, status):
        nonlocal is_speaking, attack_count, silence_count, audio_buffer, rms_history, state, interrupt_frames, START_THRESHOLD, HOLD_THRESHOLD, MIN_PEAK_RMS, INTERRUPT_RMS
        
        # 1. 通道解包与单/双发射器智能混音
        if mic_channels == 2:
            stereo = np.frombuffer(indata, dtype=np.int16).reshape(-1, 2)
            ch0 = stereo[:, 0]
            ch1 = stereo[:, 1]
            rms0 = int(np.sqrt(np.mean(ch0.astype(np.float32)**2)))
            rms1 = int(np.sqrt(np.mean(ch1.astype(np.float32)**2)))
            # 无线领夹麦智能适配：若某个声道显著更响（如只开了一个发射器），取该通道避免除以2音量损失
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

        # 启动自适应校准底噪（前 1.0 秒）
        if not is_ready_to_listen:
            calib_samples.append(rms)
            return

        bars = "▇" * min(12, rms // 30) + "░" * max(0, 12 - rms // 30)

        # 1. 扬声器播报中：防回声，需连续 3 帧大声打断
        if state == STATE_SPEAKING:
            if rms >= INTERRUPT_RMS:
                interrupt_frames += 1
                if interrupt_frames >= 3:
                    player.interrupt()
                    set_state(STATE_LISTENING)
                    is_speaking = True
                    audio_buffer = [raw_bytes]
                    rms_history = [rms]
                    silence_count = 0
                    attack_count = 0
                    log_event("USER_INTERRUPT", f"User interrupted speaking (rms={rms})")
                    sys.stdout.write(f"\r🛑 [\033[1;31m已打断播报，请继续说...\033[0m]                           \n")
                    sys.stdout.flush()
            else:
                interrupt_frames = 0
            sys.stdout.write(f"\r🗣️ [\033[1;35mGemini 正在播报...\033[0m] 音量: |{bars}| ({rms:3d}) ")
            sys.stdout.flush()
            return

        # 2. kimi-cu 正在物理执行动作
        if state == STATE_EXECUTING:
            sys.stdout.write(f"\r⚙️ [\033[1;33mkimi-cu 正在执行桌面动作...\033[0m] 音量: |{bars}| ({rms:3d}) ")
            sys.stdout.flush()
            return

        # 3. 正在思考/等待工具结果汇总中：静默防护，丢弃微小杂音，支持高声打断
        if state == STATE_THINKING:
            if rms >= INTERRUPT_RMS:
                interrupt_frames += 1
                if interrupt_frames >= 4:
                    set_state(STATE_LISTENING)
                    is_speaking = True
                    audio_buffer = [raw_bytes]
                    rms_history = [rms]
                    silence_count = 0
                    attack_count = 0
                    log_event("USER_INTERRUPT", f"User interrupted thinking state (rms={rms})")
                    sys.stdout.write(f"\r🛑 [\033[1;31m已取消等待，请重新说...\033[0m]                           \n")
                    sys.stdout.flush()
            else:
                interrupt_frames = 0
            sys.stdout.write(f"\r🧠 [\033[1;36mGemini 正在处理中...\033[0m] 音量: |{bars}| ({rms:3d}) ")
            sys.stdout.flush()
            return

        # 4. 正常空闲监听中（STATE_LISTENING）：智能双门限近场语音门控
        if not is_speaking:
            # 门控检测：必须连续 ATTACK_FRAMES 帧超过 START_THRESHOLD 才确认为近场真人起呼
            if rms >= START_THRESHOLD:
                attack_count += 1
                pre_roll.append((raw_bytes, rms))
                sys.stdout.write(f"\r🎤 [\033[1;33m检测到声音 {attack_count}/{ATTACK_FRAMES}\033[0m] 音量: |{bars}| ({rms:3d}/{START_THRESHOLD}) ")
                sys.stdout.flush()

                if attack_count >= ATTACK_FRAMES:
                    is_speaking = True
                    audio_buffer = [b for b, r in pre_roll]
                    rms_history = [r for b, r in pre_roll]
                    silence_count = 0
                    attack_count = 0
            else:
                attack_count = 0
                pre_roll.append((raw_bytes, rms))
                # 当有微弱声音但未达起呼门限时，给出友好的实时反馈
                if rms >= max(35, int(START_THRESHOLD * 0.55)):
                    sys.stdout.write(f"\r🎤 [\033[93m收音中\033[0m] 音量: |{bars}| ({rms:3d}/{START_THRESHOLD}) ")
                else:
                    sys.stdout.write(f"\r🎤 [\033[90m监听中\033[0m] 音量: |{bars}| ({rms:3d}/{START_THRESHOLD}) ")
                sys.stdout.flush()
        else:
            # 处于说话状态中：使用更宽容的 HOLD_THRESHOLD 保护句尾弱音与轻声字
            audio_buffer.append(raw_bytes)
            rms_history.append(rms)

            if rms >= HOLD_THRESHOLD:
                silence_count = 0
                dur = len(audio_buffer) * CHUNK_SIZE / MIC_RATE
                sys.stdout.write(f"\r🎤 [\033[1;32m正在说话\033[0m] 音量: |{bars}| ({rms:3d}) 已录制 {dur:.1f}s ")
                sys.stdout.flush()
            else:
                silence_count += 1
                sys.stdout.write(f"\r🎤 [\033[1;36m检测停顿 {silence_count}/{SILENCE_CHUNKS}\033[0m] 音量: |{bars}| ({rms:3d}) ")
                sys.stdout.flush()

                if silence_count >= SILENCE_CHUNKS:
                    is_speaking = False
                    full_turn = b"".join(audio_buffer)
                    dur_sec = len(full_turn) / (MIC_RATE * 2)
                    peak_rms = max(rms_history) if rms_history else 0
                    voiced_ratio = sum(1 for r in rms_history if r >= START_THRESHOLD) / max(1, len(rms_history))

                    audio_buffer = []
                    rms_history = []
                    silence_count = 0
                    attack_count = 0

                    # 智能语音质量过滤：拦截极微弱的环境底噪晃动
                    if dur_sec < MIN_DURATION or peak_rms < MIN_PEAK_RMS or voiced_ratio < MIN_VOICED_RATIO:
                        log_event("NOISE_REJECTED", f"dur={dur_sec:.2f}s, peak={peak_rms}, voiced={voiced_ratio:.2f}, threshold={START_THRESHOLD}")
                        sys.stdout.write(f"\r🔇 [\033[90m已过滤微弱背景声 (峰值:{peak_rms}/{MIN_PEAK_RMS}，时长:{dur_sec:.1f}s)\033[0m]                       \n")
                        sys.stdout.flush()
                        return

                    # 确认为近场清晰指令，投递给 Gemini 并转为 THINKING
                    set_state(STATE_THINKING)
                    log_event("USER_VOICE", f"Captured voice of {dur_sec:.2f}s (peak={peak_rms}, {len(full_turn)} bytes), sending to Gemini")
                    sys.stdout.write(f"\r⚡ [\033[1;33m正在发送语音至 Gemini 3.8 Live...\033[0m]                       \n")
                    sys.stdout.flush()
                    loop.call_soon_threadsafe(turn_queue.put_nowait, full_turn)

    # 建立全双工连接
    async with client.aio.live.connect(model=selected_model, config=config) as session:
        log_event("SESSION", f"Gemini Live session connected ({selected_model})")

        mic_stream = sd.RawInputStream(
            samplerate=MIC_RATE,
            channels=mic_channels,
            dtype="int16",
            blocksize=CHUNK_SIZE,
            device=mic_idx,
            callback=mic_callback
        )
        mic_stream.start()

        # 采集 1.2 秒现场底噪（丢弃前 0.3 秒硬件开关冲击声），精确计算本底噪声
        await asyncio.sleep(1.2)
        if not user_threshold and calib_samples:
            warm_samples = calib_samples[5:] if len(calib_samples) > 8 else calib_samples
            noise_median = int(np.median(warm_samples))
            noise_p75 = int(np.percentile(warm_samples, 75))
            noise_mean = int(np.mean(warm_samples))
            # 使用 P75 作为稳健底噪估计（彻底抵御开机碰触尖峰，真实反映底噪）
            base_noise = noise_p75
            START_THRESHOLD = max(65, min(160, int(base_noise * 1.7 + 25)))
            HOLD_THRESHOLD = max(35, min(90, int(base_noise * 1.1 + 10)))
            MIN_PEAK_RMS = max(75, int(START_THRESHOLD * 1.15))
            INTERRUPT_RMS = max(160, int(START_THRESHOLD * 1.7))
            log_event("CALIBRATION", f"Noise floor median={noise_median}, mean={noise_mean}, p75={noise_p75}, start_threshold={START_THRESHOLD}, hold_threshold={HOLD_THRESHOLD}, min_peak={MIN_PEAK_RMS}")

        is_ready_to_listen = True

        sys.stdout.write(f"\r🟢 [\033[1;32m连接就绪，请直接对麦克风说话\033[0m] (起呼门限: {START_THRESHOLD}, 维持门限: {HOLD_THRESHOLD})                 \n")
        sys.stdout.flush()

        # 【保姆式看门狗自愈】：如果在非空闲状态且扬声器未在播放，超过 25.0 秒毫无响应，自动恢复空闲监听
        async def watchdog_loop():
            while not shutdown_event.is_set() and not reconnect_event.is_set():
                await asyncio.sleep(1.0)
                if state in [STATE_THINKING, STATE_EXECUTING] and not player.is_busy():
                    idle_sec = time.time() - state_start_time
                    if idle_sec > 25.0:
                        log_event("WATCHDOG_TIMEOUT", f"Server/tool unresponsive for {idle_sec:.1f}s, recovering to LISTENING")
                        sys.stdout.write("\n⚠️ [\033[1;33m响应超时，已自动恢复待命状态，请重新说话...\033[0m]\n")
                        sys.stdout.flush()
                        set_state(STATE_LISTENING)

        async def send_loop():
            try:
                while not shutdown_event.is_set() and not reconnect_event.is_set():
                    pcm_turn = await turn_queue.get()
                    await session.send_client_content(
                        turns=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_bytes(
                                        data=pcm_turn,
                                        mime_type="audio/pcm;rate=16000"
                                    )
                                ]
                            )
                        ],
                        turn_complete=True
                    )
            except asyncio.CancelledError:
                pass
            except Exception as e:
                log_event("SEND_ERROR", str(e))

        async def recv_loop():
            nonlocal state, has_active_tool, waiting_tool_summary
            try:
                while not shutdown_event.is_set() and not reconnect_event.is_set():
                    response = await session._receive()
                    if not response:
                        break

                    # 1. 检查打断
                    if response.server_content and response.server_content.interrupted:
                        player.interrupt()
                        set_state(STATE_LISTENING)
                        has_active_tool = False
                        waiting_tool_summary = False
                        log_event("SERVER_INTERRUPT", "Server reported interrupted")

                    # 2. 检查模型语音或文字
                    if response.server_content and response.server_content.model_turn:
                        # 收到模型实际返回（文字或音频），表明工具结果总结正在输出
                        waiting_tool_summary = False
                        for part in response.server_content.model_turn.parts:
                            if part.text:
                                if getattr(part, "thought", False):
                                    log_event("MODEL_THOUGHT", part.text.strip())
                                    sys.stdout.write(f"\033[90m💭 {part.text}\033[0m")
                                else:
                                    log_event("MODEL_TEXT", part.text.strip())
                                    sys.stdout.write(part.text)
                                sys.stdout.flush()
                            if part.inline_data:
                                set_state(STATE_SPEAKING)
                                player.write(part.inline_data.data)

                    # 3. 检查工具调用请求（100% 走 kimi-cu MCP）
                    if response.tool_call:
                        has_active_tool = True
                        waiting_tool_summary = False
                        set_state(STATE_EXECUTING)
                        function_responses = []
                        for call in response.tool_call.function_calls:
                            func_name = call.name
                            func_args = call.args or {}
                            log_event("MCP_CALL", f"Calling {func_name} with args: {func_args}")
                            print(f"\n🛠️  [kimi-cu 动作] \033[1;33m{func_name}\033[0m({func_args})")

                            if func_name in ["press_key", "type_text", "click"]:
                                if not func_args.get("app") and not func_args.get("pid"):
                                    func_args["app"] = "com.apple.finder"

                            t_start = time.time()
                            try:
                                mcp_res = await mcp_session.call_tool(func_name, func_args)
                                cost_ms = int((time.time() - t_start) * 1000)
                                texts = []
                                for item in mcp_res.content:
                                    if hasattr(item, "text") and item.text:
                                        texts.append(item.text)
                                    elif hasattr(item, "data"):
                                        texts.append("[截图像素数据已捕获]")
                                raw_text = "\n".join(texts) if texts else "ok"
                                
                                # 高密度精简提取，杜绝超长文本阻塞 WebSocket 音频推理
                                res_text = clean_ax_text(raw_text, max_chars=1800)
                                log_event("MCP_RESULT", f"{func_name} ({cost_ms}ms) raw={len(raw_text)} chars, cleaned={len(res_text)} chars")
                                print(f"✨ [kimi-cu 完成] {func_name} ({cost_ms}ms, 提取有效内容 {len(res_text)} 字符)")
                            except Exception as err:
                                res_text = f"Error: {err}"
                                log_event("MCP_ERROR", f"{func_name} failed: {err}")
                                print(f"❌ [kimi-cu 失败] {err}")

                            function_responses.append(
                                types.FunctionResponse(
                                    name=func_name,
                                    id=call.id,
                                    response={"result": res_text}
                                )
                            )

                        await session.send_tool_response(function_responses=function_responses)
                        log_event("TOOL_RESPONSE_SENT", f"Sent response for {len(function_responses)} calls, waiting for Gemini summary")
                        # 工具发回后，标记等待口头总结，切为 THINKING，严禁被并发 turn_complete 误杀
                        has_active_tool = False
                        waiting_tool_summary = True
                        set_state(STATE_THINKING)

                    # 4. 轮次终结：仅当没有未完成的工具调用、且不在等待工具总结、且播放器彻底播完后才回到 LISTENING
                    if response.server_content and response.server_content.turn_complete:
                        if not response.tool_call and not has_active_tool and not waiting_tool_summary:
                            async def wait_for_playback_done():
                                wait_start = time.time()
                                while player.is_busy() and (time.time() - wait_start < 15.0):
                                    await asyncio.sleep(0.05)
                                await asyncio.sleep(0.1)
                                set_state(STATE_LISTENING)
                                log_event("TURN_COMPLETE", "Turn fully finished and playback done, now LISTENING")
                                sys.stdout.write("\n🟢 [\033[1;32m就绪，请说下一句指令...\033[0m]\n")
                                sys.stdout.flush()

                            asyncio.create_task(wait_for_playback_done())

            except asyncio.CancelledError:
                pass
            except Exception as e:
                log_event("RECV_ERROR", str(e))
            finally:
                set_state(STATE_LISTENING)

        task_send = asyncio.create_task(send_loop())
        task_recv = asyncio.create_task(recv_loop())
        task_watchdog = asyncio.create_task(watchdog_loop())

        try:
            await asyncio.wait([task_send, task_recv, task_watchdog], return_when=asyncio.FIRST_COMPLETED)
        finally:
            task_send.cancel()
            task_recv.cancel()
            task_watchdog.cancel()
            try:
                mic_stream.stop()
                mic_stream.close()
            except Exception:
                pass


async def main():
    parser = argparse.ArgumentParser(description="Gemini 3.8 Live + kimi-cu 实时语音电脑操作管家 (纯原生 kimi-cu MCP)")
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
        help="近场说话起呼门限 RMS（默认自动根据底噪自适应，通常 80~150；调高更防杂音，调低更灵敏）"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=os.environ.get("GEMINI_VOICE", "Aoede"),
        help="语音音色 (可选: Aoede, Puck, Charon, Kore, Fenrir)"
    )
    args = parser.parse_args()

    selected_model = MODEL_THINKING if args.thinking else args.model
    mic_idx, mic_name, mic_channels = find_audio_devices(args.mic)
    kimi_cu_path = os.environ.get("KIMI_CU_PATH", "/Applications/KimiCU.app/Contents/MacOS/kimi-cu")

    log_event("STARTUP", f"Starting service with model={selected_model}, mic={mic_name} (ch={mic_channels}), threshold={args.threshold}, voice={args.voice}")

    print("=" * 68)
    print("🎙️   Gemini 3.8 Live + kimi-cu 纯原生 MCP 实时语音桌面操作管家")
    print(f"🤖  当前模型: \033[1;36m{selected_model}\033[0m")
    print(f"🎤  输入麦克风: \033[1;32m[{mic_idx}] {mic_name} ({mic_channels}通道)\033[0m")
    if args.threshold:
        print(f"🎯  降噪门限: \033[1;33m固定近场门限 RMS={args.threshold}\033[0m (拦截远场闲聊与环境杂音)")
    else:
        print(f"🎯  降噪门限: \033[1;32m智能自适应近场门控\033[0m (开机自动采样底噪对齐)")
    print(f"🗣️  当前音色: \033[1;35m{args.voice}\033[0m (可选: Aoede, Puck, Charon, Kore, Fenrir)")
    print(f"📋  日志文件: \033[1;34m{LOG_FILE}\033[0m")
    if "thinking" in selected_model:
        print("🧠  思考模式: \033[1;35m已启用 Extended Thinking (深度推理 + 实时语音交互)\033[0m")
    else:
        print("⚡  响应模式: \033[1;33mUltra-Low Latency (极速交互)\033[0m")
    print("=" * 68)

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("\n未检测到 GEMINI_API_KEY 环境变量。")
        api_key = input("👉 请输入你的 Google Gemini API Key: ").strip()
        if not api_key:
            print("❌ 必须提供 API Key 才能启动。退出。")
            return
        os.environ["GEMINI_API_KEY"] = api_key
        with open(env_path, "w", encoding="utf-8") as f:
            f.write(f"GEMINI_API_KEY={api_key}\n")
            f.write(f"GEMINI_LIVE_MODEL={selected_model}\n")
            f.write(f"PREFER_MIC={args.mic}\n")
            if args.threshold:
                f.write(f"MIC_THRESHOLD={args.threshold}\n")
            f.write(f"GEMINI_VOICE={args.voice}\n")

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

                tool_names = [t.name for t in tools_resp.tools]
                print(f"✅ 成功加载 {len(gemini_functions)} 个 kimi-cu 原生桌面控制工具:")
                print("   " + ", ".join(tool_names))
                print(f"\n[2/3] 正在建立 Gemini Live 全双工连接 ({selected_model})...")
                print(f"[3/3] 🟢 实时语音管家已就绪！")
                print(f"💡 对着 \033[1;32m{mic_name}\033[0m 说话，停顿后自动执行。按 Ctrl+C 退出。\n" + "-" * 68)

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
                            user_threshold=args.threshold
                        )
                    except asyncio.CancelledError:
                        break
                    except Exception as e:
                        log_event("RECONNECT", f"Connection dropped, retrying: {e}")
                        print(f"\n⚠️ [连接断开，正在自动重连]: {e}", file=sys.stderr)
                        await asyncio.sleep(1.0)

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
