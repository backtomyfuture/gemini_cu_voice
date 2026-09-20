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
"""
自动化多层测试套件 (Test Suite for Gemini Live CU Voice Assistant)
用于在人工语音实测前，对底层工具原子、大模型意图决策、多轮闭环防死循环以及音频硬件进行全方位自动化验证。
"""
import os
import sys
import time
import json
import asyncio
import argparse
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

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

from google import genai
from google.genai import types

from gemini_live_cu import (
    launch_mac_app,
    format_tool_result,
    find_audio_devices,
    SYSTEM_INSTRUCTION
)
from ego_browser_client import (
    browser_open,
    browser_search,
    browser_get_content,
    browser_scroll,
    get_browser_function_declarations
)
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# 颜色控制
GREEN = "\033[1;32m"
RED = "\033[1;31m"
YELLOW = "\033[1;33m"
CYAN = "\033[1;36m"
BOLD = "\033[1m"
RESET = "\033[0m"


class TestReport:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.results = []

    def record(self, layer: str, test_name: str, success: bool, detail: str = "", cost_ms: float = 0):
        if success:
            self.passed += 1
            status_str = f"{GREEN}✓ PASS{RESET}"
        else:
            self.failed += 1
            status_str = f"{RED}✗ FAIL{RESET}"
        
        self.results.append({
            "layer": layer,
            "name": test_name,
            "success": success,
            "detail": detail,
            "cost_ms": cost_ms
        })
        print(f"  [{status_str}] {BOLD}{test_name}{RESET} ({cost_ms:.0f}ms)")
        if detail:
            for line in detail.strip().splitlines()[:3]:
                print(f"         \033[90m{line}\033[0m")

    def print_summary(self):
        total = self.passed + self.failed + self.skipped
        print("\n" + "=" * 70)
        print(f"{BOLD}📊 测试报告汇总 (Test Execution Summary){RESET}")
        print("=" * 70)
        print(f"总测试项: {total} | 通过: {GREEN}{self.passed}{RESET} | 失败: {RED}{self.failed}{RESET} | 跳过: {YELLOW}{self.skipped}{RESET}")
        if self.failed == 0:
            print(f"\n🎉 {GREEN}{BOLD}所有测试全部通过！系统状态健康，已具备安全实测条件。{RESET}\n")
        else:
            print(f"\n⚠️ {RED}{BOLD}存在 {self.failed} 项测试未通过，请检查上方日志并进行针对性排查。{RESET}\n")


async def get_all_tool_declarations(mcp_session=None):
    """获取全量工具声明（open_app + kimi-cu + ego-browser）"""
    tools = []
    
    # 1. open_app
    tools.append(types.FunctionDeclaration(
        name="open_app",
        description="在 macOS 上启动或前台激活任何应用程序（如 计算器, 备忘录, 微信, 音乐, Safari 等）。",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "应用程序名称，支持中文或英文"},
                "bundle_id": {"type": "string", "description": "可选。Bundle ID"}
            },
            "required": ["name"]
        }
    ))

    # 2. ego-browser
    tools.extend(get_browser_function_declarations())

    # 3. kimi-cu
    if mcp_session:
        try:
            mcp_tools = await mcp_session.list_tools()
            for t in mcp_tools.tools:
                tools.append(types.FunctionDeclaration(
                    name=t.name,
                    description=t.description or "",
                    parameters=t.input_schema or {"type": "object", "properties": {}}
                ))
        except Exception as e:
            print(f"Warning: Could not fetch mcp tools: {e}")

    return tools


# ==============================================================================
# Layer 1: 底层工具原子执行测试 (Tool Primitives Boundary)
# ==============================================================================
async def test_layer_1(report: TestReport, mcp_session):
    print(f"\n{CYAN}{BOLD}【Layer 1】底层工具原子执行基座测试{RESET}")
    print("-" * 70)

    # 1.1 open_app 原生应用打开测试
    t0 = time.time()
    res = launch_mac_app("计算器", "com.apple.calculator")
    cost = (time.time() - t0) * 1000
    ok = "成功打开" in res
    report.record("Layer 1", "原生工具: launch_mac_app('计算器')", ok, res, cost)

    # 1.2 kimi-cu list_apps 测试
    t0 = time.time()
    try:
        mcp_res = await mcp_session.call_tool("list_apps", {})
        cost = (time.time() - t0) * 1000
        raw = "\n".join([i.text for i in mcp_res.content if hasattr(i, "text")])
        formatted = format_tool_result("list_apps", raw)
        ok = len(formatted) > 20 and ("com.apple" in formatted or "Calculator" in formatted or "Finder" in formatted)
        report.record("Layer 1", "kimi-cu: list_apps()", ok, f"返回 {len(formatted)} 字符: {formatted[:60]}...", cost)
    except Exception as e:
        cost = (time.time() - t0) * 1000
        report.record("Layer 1", "kimi-cu: list_apps()", False, str(e), cost)

    # 1.3 ego-browser open 网页加载测试
    t0 = time.time()
    try:
        res = await browser_open("https://example.com", max_chars=300)
        cost = (time.time() - t0) * 1000
        ok = "Example Domain" in res and "【页面标题】" in res
        report.record("Layer 1", "ego-browser: browser_open('https://example.com')", ok, res[:100], cost)
    except Exception as e:
        cost = (time.time() - t0) * 1000
        report.record("Layer 1", "ego-browser: browser_open", False, str(e), cost)

    # 1.4 ego-browser search 搜索能力测试
    t0 = time.time()
    try:
        res = await browser_search("特斯拉 Roadster 2026", max_chars=300)
        cost = (time.time() - t0) * 1000
        ok = "【页面标题】" in res and len(res) > 50
        report.record("Layer 1", "ego-browser: browser_search('特斯拉 Roadster 2026')", ok, res[:100], cost)
    except Exception as e:
        cost = (time.time() - t0) * 1000
        report.record("Layer 1", "ego-browser: browser_search", False, str(e), cost)

    # 1.5 ego-browser scroll 页面滚动测试
    t0 = time.time()
    try:
        res = await browser_scroll("down")
        cost = (time.time() - t0) * 1000
        ok = "成功" in res or "已向" in res
        report.record("Layer 1", "ego-browser: browser_scroll('down')", ok, res, cost)
    except Exception as e:
        cost = (time.time() - t0) * 1000
        report.record("Layer 1", "ego-browser: browser_scroll", False, str(e), cost)


# ==============================================================================
# Layer 2: 大模型意图决策与工具路由评测 (LLM Decision Boundary)
# ==============================================================================
async def test_layer_2(report: TestReport, client: genai.Client, all_tools):
    print(f"\n{CYAN}{BOLD}【Layer 2】大模型意图决策与工具调用评测{RESET}")
    print("-" * 70)

    eval_model = os.environ.get("EVAL_MODEL", "gemini-3.6-flash")

    test_cases = [
        {
            "query": "帮我打开计算器",
            "expected_tool": "open_app",
            "validator": lambda args: "计算器" in args.get("name", "") or "Calculator" in args.get("name", "")
        },
        {
            "query": "在浏览器里帮我搜一下特斯拉 Roadster 2026 最新消息",
            "expected_tool": "browser_search",
            "validator": lambda args: "特斯拉" in args.get("query", "")
        },
        {
            "query": "帮我看看现在电脑里正在运行什么软件",
            "expected_tool": "list_apps",
            "validator": lambda args: True
        },
        {
            "query": "早上好啊，今天天气真不错，心情挺好的！",
            "expected_tool": None,  # 闲聊不应触发任何工具调用
            "validator": lambda args: True
        }
    ]

    for tc in test_cases:
        query = tc["query"]
        expected_tool = tc["expected_tool"]
        t0 = time.time()
        try:
            resp = await client.aio.models.generate_content(
                model=eval_model,
                contents=query,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[types.Tool(function_declarations=all_tools)],
                    temperature=0.1
                )
            )
            cost = (time.time() - t0) * 1000

            func_calls = resp.function_calls or []
            if expected_tool is None:
                # 闲聊：不应调用工具
                if len(func_calls) == 0 and resp.text:
                    report.record("Layer 2", f"意图识别: '{query}' -> 纯文本回复", True, f"回复: {resp.text.strip()[:60]}", cost)
                else:
                    called = [f.name for f in func_calls]
                    report.record("Layer 2", f"意图识别: '{query}' -> 纯文本回复", False, f"错误调用了工具: {called}", cost)
            else:
                # 必须命中对应工具
                matched = [f for f in func_calls if f.name == expected_tool]
                if matched and tc["validator"](matched[0].args):
                    report.record("Layer 2", f"意图识别: '{query}' -> {expected_tool}", True, f"参数: {matched[0].args}", cost)
                else:
                    called = [(f.name, f.args) for f in func_calls]
                    report.record("Layer 2", f"意图识别: '{query}' -> {expected_tool}", False, f"实际调用: {called}", cost)

        except Exception as e:
            cost = (time.time() - t0) * 1000
            report.record("Layer 2", f"意图识别: '{query}'", False, str(e), cost)


# ==============================================================================
# Layer 3: 端到端 Mock 闭环与防死循环评测 (End-to-End & Anti-Loop Boundary)
# ==============================================================================
async def test_layer_3(report: TestReport, client: genai.Client, all_tools):
    print(f"\n{CYAN}{BOLD}【Layer 3】端到端两轮闭环与防死循环评测{RESET}")
    print("-" * 70)

    eval_model = os.environ.get("EVAL_MODEL", "gemini-3.6-flash")

    # Case 3.1: 打开计算器 -> 收到结果 -> 必须总结收口，严禁死循环重复调用
    t0 = time.time()
    try:
        user_turn = "帮我打开计算器"
        # 轮次 1: 用户发言 -> 模型产生 tool call
        r1 = await client.aio.models.generate_content(
            model=eval_model,
            contents=user_turn,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[types.Tool(function_declarations=all_tools)],
                temperature=0.1
            )
        )
        call = (r1.function_calls or [None])[0]
        if not call or call.name != "open_app":
            report.record("Layer 3", "E2E闭环: 打开计算器首轮调用", False, f"未触发 open_app: {r1.function_calls}")
        else:
            # 轮次 2: 模拟回传工具执行成功结果
            tool_result_content = "成功打开并激活应用: 计算器"
            history = [
                types.Content(role="user", parts=[types.Part.from_text(text=user_turn)]),
                r1.candidates[0].content,
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name="open_app",
                            response={"result": tool_result_content}
                        )
                    ]
                )
            ]
            r2 = await client.aio.models.generate_content(
                model=eval_model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[types.Tool(function_declarations=all_tools)],
                    temperature=0.1
                )
            )
            cost = (time.time() - t0) * 1000
            
            # 核心断言：模型必须给出自然语言总结，并且绝对不能再次下发工具调用（防死循环）
            has_loop_call = bool(r2.function_calls)
            has_verbal_summary = bool(r2.text and len(r2.text.strip()) > 0)
            
            if not has_loop_call and has_verbal_summary:
                report.record("Layer 3", "E2E闭环: 打开计算器结果收口 (防死循环)", True, f"口语总结: {r2.text.strip()}", cost)
            else:
                detail = f"死循环重复调用: {r2.function_calls}" if has_loop_call else "无口语总结"
                report.record("Layer 3", "E2E闭环: 打开计算器结果收口 (防死循环)", False, detail, cost)
    except Exception as e:
        cost = (time.time() - t0) * 1000
        report.record("Layer 3", "E2E闭环: 打开计算器", False, str(e), cost)

    # Case 3.2: 搜索内容 -> 模拟回传网页要点 -> 验证归纳回答
    t0 = time.time()
    try:
        user_turn = "在浏览器搜一下特斯拉 Roadster 2026"
        r1 = await client.aio.models.generate_content(
            model=eval_model,
            contents=user_turn,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[types.Tool(function_declarations=all_tools)],
                temperature=0.1
            )
        )
        call = (r1.function_calls or [None])[0]
        if not call or call.name != "browser_search":
            report.record("Layer 3", "E2E闭环: 浏览器搜索首轮调用", False, f"未触发 browser_search: {r1.function_calls}")
        else:
            mock_web_text = "【页面标题】: 特斯拉新一代 Roadster 最新进展\n【URL】: https://baidu.com/s?wd=...\n【正文要点】: 特斯拉宣布新一代 Roadster 跑车计划于 2026 年量产交付，百公里加速将在1秒以内，采用 Space X 冷气推力器技术。"
            history = [
                types.Content(role="user", parts=[types.Part.from_text(text=user_turn)]),
                r1.candidates[0].content,
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_function_response(
                            name="browser_search",
                            response={"result": mock_web_text}
                        )
                    ]
                )
            ]
            r2 = await client.aio.models.generate_content(
                model=eval_model,
                contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    tools=[types.Tool(function_declarations=all_tools)],
                    temperature=0.1
                )
            )
            cost = (time.time() - t0) * 1000
            
            has_loop_call = bool(r2.function_calls)
            has_summary = bool(r2.text and ("2026" in r2.text or "Roadster" in r2.text or "交付" in r2.text))
            
            if not has_loop_call and has_summary:
                report.record("Layer 3", "E2E闭环: 浏览器搜索内容提炼总结", True, f"模型回答: {r2.text.strip()[:80]}...", cost)
            else:
                report.record("Layer 3", "E2E闭环: 浏览器搜索内容提炼总结", False, f"死循环: {r2.function_calls}, 文本: {r2.text}", cost)
    except Exception as e:
        cost = (time.time() - t0) * 1000
        report.record("Layer 3", "E2E闭环: 浏览器搜索", False, str(e), cost)


# ==============================================================================
# Layer 4: 音频硬件与 VAD 底噪健康检查 (Audio & VAD Health Boundary)
# ==============================================================================
def test_layer_4(report: TestReport, prefer_mic="Wireless Mic Rx"):
    print(f"\n{CYAN}{BOLD}【Layer 4】音频硬件与近场 VAD 门控健康检查{RESET}")
    print("-" * 70)

    import sounddevice as sd

    t0 = time.time()
    try:
        mic_idx, mic_name, mic_channels = find_audio_devices(prefer_mic)
        report.record("Layer 4", f"设备识别: [{mic_idx}] {mic_name} ({mic_channels}通道)", True, "设备正常就绪")

        # 录制 1.0 秒环境声样本
        sample_rate = 16000
        chunk_size = 1024
        samples = []

        def audio_cb(indata, frames, time_info, status):
            if mic_channels == 2:
                stereo = np.frombuffer(indata, dtype=np.int16).reshape(-1, 2)
                ch0 = stereo[:, 0]
                ch1 = stereo[:, 1]
                rms0 = int(np.sqrt(np.mean(ch0.astype(np.float32)**2)))
                rms1 = int(np.sqrt(np.mean(ch1.astype(np.float32)**2)))
                rms = max(rms0, rms1)
            else:
                mono = np.frombuffer(indata, dtype=np.int16)
                rms = int(np.sqrt(np.mean(mono.astype(np.float32)**2)))
            samples.append(rms)

        stream = sd.RawInputStream(
            samplerate=sample_rate,
            channels=mic_channels,
            dtype="int16",
            blocksize=chunk_size,
            device=mic_idx,
            callback=audio_cb
        )
        with stream:
            time.sleep(1.0)

        cost = (time.time() - t0) * 1000

        # 舍弃前 0.3 秒开机冲击
        warm = samples[5:] if len(samples) > 8 else samples
        if not warm:
            report.record("Layer 4", "麦克风采集样本", False, "未能采集到有效音频样本", cost)
            return

        p75 = int(np.percentile(warm, 75))
        median = int(np.median(warm))
        computed_start = max(65, min(160, int(p75 * 1.7 + 25)))
        computed_hold = max(35, min(90, int(p75 * 1.1 + 10)))

        is_healthy = 10 <= p75 <= 150
        detail = f"底噪 P75={p75}, Median={median} -> 自适应起呼门限={computed_start}, 维持门限={computed_hold}"
        report.record("Layer 4", "自适应门限健康度诊断", is_healthy, detail, cost)

    except Exception as e:
        cost = (time.time() - t0) * 1000
        report.record("Layer 4", "音频硬件诊断", False, str(e), cost)


# ==============================================================================
# 主入口
# ==============================================================================
async def main():
    parser = argparse.ArgumentParser(description="语音电脑管家全层级自动化测试套件")
    parser.add_argument("--layer", type=int, choices=[1, 2, 3, 4], help="仅运行指定层级测试")
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}🧪  Gemini Live CU + Ego Browser 自动化测试验证套件{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")

    report = TestReport()
    api_key = os.environ.get("GEMINI_API_KEY")
    kimi_cu_path = os.environ.get("KIMI_CU_PATH", "/Applications/KimiCU.app/Contents/MacOS/kimi-cu")

    if not api_key:
        print(f"{RED}错误: 未检测到 GEMINI_API_KEY 环境变量{RESET}")
        return

    client = genai.Client(api_key=api_key)

    # 启动 MCP 连接 kimi-cu
    mcp_params = StdioServerParameters(
        command=kimi_cu_path,
        args=["mcp", "-s", "user"],
        env=os.environ.copy()
    )

    async with stdio_client(mcp_params) as (mcp_read, mcp_write):
        async with ClientSession(mcp_read, mcp_write) as mcp_session:
            await mcp_session.initialize()
            all_tools = await get_all_tool_declarations(mcp_session)

            if args.layer is None or args.layer == 1:
                await test_layer_1(report, mcp_session)

            if args.layer is None or args.layer == 2:
                await test_layer_2(report, client, all_tools)

            if args.layer is None or args.layer == 3:
                await test_layer_3(report, client, all_tools)

            if args.layer is None or args.layer == 4:
                test_layer_4(report)

    report.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
