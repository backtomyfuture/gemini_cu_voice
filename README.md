# Gemini 3.8 Live + kimi-cu / Ego Lite 实时语音桌面操作管家

这是一个轻量级、超低延迟、全双工的 macOS 桌面实时语音操作管家。

通过 **Google Gemini 3.8 Multimodal Live API**，以全双工双向实时流式音频（HybridVAD 模式）与你交谈，并协同三大执行引擎实现多维度自动化控制：
1. **macOS 原生应用管理** (`open_app`)：极速启动与前台置顶 macOS 应用。
2. **kimi-cu 原生桌面控制**（11 个 MCP 原生工具）：无障碍控件树识别、桌面点击、文本键入与按键操作。
3. **Ego Lite 极速浏览器自动化**（6 个专有工具）：网页搜索、正文结构化提炼、页面候选编号操作清单与抗漂移精准点击。

---

## 核心特性

- ⚡ **超低延迟全双工语音（HybridVAD）**：本地近场门控 + 持续音频流分块发送 + `audio_stream_end`，端到端延迟低至数百毫秒。
- 🗣️ **毫秒级实时打断（Barge-in）**：在 AI 播报或工具执行时开口，瞬间中断播报与动作。
- 🎙️ **智能双门限近场语音门控**：
  - 自动过滤远场他人闲聊与环境杂音（敲键盘、关门、脚步声等），绝不误触。
  - 开机 1.2 秒自适应底噪校准（基准起呼门限约 65~160 RMS），支持通过 `--threshold` 自由调节门限。
- 🛡️ **安全执行层与防死循环治理**：
  - 严格拦截破坏性高危按键与恶意 Prompt 注入。
  - 单轮工具调用预算控制（默认上限 8 次）与相同参数重复调用自动去重拦截。
- 🧠 **会话持久化与断线无缝恢复**：
  - 支持官方 `session_resumption` 句柄恢复与滑动窗口上下文压缩（`context_window_compression`）。
  - 本地记忆池（`ConversationMemory`）支持多轮记忆累积与冷启动回灌。
- 📦 **免配环境**：借助 `uv` 与 `uv.lock` 锁定所有依赖版本，一键即跑。

---

## 快速运行

### 方式 1：直接运行启动脚本（推荐）
```bash
/Users/jarod/Documents/gemini_cu_voice/run.sh
```

- **自定义近场起呼门限**（数值越大越只认贴近大声说，默认开机自适应对齐，通常 80~150）：
  ```bash
  /Users/jarod/Documents/gemini_cu_voice/run.sh --threshold 130
  ```
- **开启深度思考模式**（复杂操作更稳健）：
  ```bash
  /Users/jarod/Documents/gemini_cu_voice/run.sh --thinking
  ```

### 方式 2：使用 uv 手动启动
```bash
cd /Users/jarod/Documents/gemini_cu_voice
uv run gemini_live_cu.py
```

首次运行时，如果在环境中未检测到 `GEMINI_API_KEY`，终端会提示你输入，并自动保存到 `.env` 文件中。

---

## 常用测试指令

佩戴麦克风（如 Wireless Mic Rx）直接说话：
- *"帮我看看当前屏幕上打开了什么软件"*
- *"帮我把浏览器打开放到屏幕上"*
- *"在地址栏输入 ithome.com 并回车"*
- *"总结一下当前网页的内容"*
- *"帮我打开微信"*
