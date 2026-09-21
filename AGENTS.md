# Agent Guidelines for `gemini_cu_voice`

## Project Overview

Gemini 3.8 Live + kimi-cu / Ego Lite 实时语音桌面操作管家。
基于 Google Gemini 3.8 Multimodal Live API，以全双工流式双向音频（HybridVAD）与用户实时交互，协同三大执行引擎实现 macOS 桌面控制与自动化：
1. **macOS 原生应用管理** (`open_app`)
2. **kimi-cu 原生桌面控制**（MCP 控件树识别、点击、键入与按键）
3. **Ego Lite 浏览器自动化**（搜索、网页结构化提取、候选清单与防漂移点击）

核心组件：
- [gemini_live_cu.py](file:///Users/jarod/Documents/gemini_cu_voice/gemini_live_cu.py): 主程序与全双工 WebSocket 事件循环、Barge-in 打断流控及会话恢复。
- [tool_policy.py](file:///Users/jarod/Documents/gemini_cu_voice/tool_policy.py): 安全策略层、高危按键阻断、单轮预算上限与调用去重治理。
- [ego_browser_client.py](file:///Users/jarod/Documents/gemini_cu_voice/ego_browser_client.py): Ego Lite 浏览器控制客户端。
- [test_suite.py](file:///Users/jarod/Documents/gemini_cu_voice/test_suite.py): 工业级 Layer 0~2 自动化测试套件（支持真实 PCM 音频全链路闭环测试）。
- [run.sh](file:///Users/jarod/Documents/gemini_cu_voice/run.sh): 快捷启动脚本。

## Development & Test Commands

使用 `uv` 统一管理 Python 环境与锁定依赖：

```bash
# 启动主程序
./run.sh
# 或者通过 uv 运行
uv run --locked python gemini_live_cu.py

# 运行自动化测试套件 (Layer 0 ~ Layer 2)
uv run python test_suite.py

# 运行特定测试层级或用例
uv run python test_suite.py --layer 0    # Layer 0: 安全策略与防死循环单测 (42项)
uv run python test_suite.py --layer 1    # Layer 1: 音频硬件识别与近场 VAD 门控健康检查
uv run python test_suite.py --layer 2    # Layer 2: 真实 PCM 语音驱动全链路应用深度闭环评测 (gemini-3.8-live)
uv run python test_suite.py --case calc  # 指定单用例真实语音测试 (calc, word, outlook, browser, notes, apps)
```

## Agent skills

### Issue tracker

GitHub issues, using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The default five-role vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context (`CONTEXT.md` at root). See `docs/agents/domain.md`.
