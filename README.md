# Gemini 3.8 Live + kimi-cu / Ego Lite 实时语音桌面操作管家

这是一个轻量级、超低延迟、全双工的 macOS 桌面实时语音操作管家。

通过 **Google Gemini 3.8 Multimodal Live API**，以全双工双向实时流式音频（HybridVAD 模式）与你交谈，并协同三大执行引擎实现多维度自动化控制：
1. **macOS 原生应用管理** (`open_app`)：极速启动与前台置顶 macOS 应用。
2. **kimi-cu 原生桌面控制**（11 个 MCP 原生工具）：无障碍控件树识别、桌面点击、文本键入与按键操作。
3. **Ego Lite 极速浏览器自动化**（6 个专有工具）：网页搜索、正文结构化提炼、页面候选编号操作清单与抗漂移精准点击。

---

## 核心特性

- ⚡ **超低延迟全双工语音（HybridVAD）**：本地近场门控 + 持续音频流分块发送 + `audio_stream_end`，端到端延迟低至数百毫秒。
- 🗣️ **毫秒级实时打断（Barge-in）与生命周期隔离**：
  - 扬声器播报、桌面动作执行、思考等待均可秒级打断。
  - `TurnController` 集中托管单调递增 `turn_id`，打断与新一轮发言均生成全新有效令牌，彻底杜绝打断后新指令失效。
  - 异步工具执行任务与接收循环解耦，打断时协同自动 `cancel()` 工具任务；迟到的播放完成回调安全丢弃，不污染新轮次。
- 🎙️ **智能双门限近场语音门控**：
  - 自动过滤远场他人闲聊与环境杂音（敲键盘、关门、脚步声等），绝不误触。
  - 开机 1.2 秒自适应底噪校准（基准起呼门限约 65~160 RMS），支持通过 `--threshold` 自由调节门限。
- 🛡️ **安全执行层与防死循环治理**：
  - 严格拦截破坏性高危按键与恶意 Prompt 注入。
  - 单轮工具调用预算控制（默认上限 8 次）与相同参数重复调用自动去重拦截。
  - 浏览器子进程超时强杀清理，严防僵尸进程；导航错误严格报错，杜绝幽灵旧页面误判。
- 🧠 **会话持久化与断线无缝恢复**：
  - 支持官方 `session_resumption` 句柄恢复与滑动窗口上下文压缩（`context_window_compression`）。
  - 会话异常时采用指数退避机制避免 0ms 热循环；当 Handle 失效时自动清空并优雅降级为本地记忆池（`ConversationMemory`）历史回灌。
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
uv run --locked python gemini_live_cu.py
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

---

## 自动化测试套件（深度闭环与真实 PCM 语音驱动）

本项目配备了工业级多层级自动化测试套件 `test_suite.py`，**支持使用真实的 16kHz 16-bit PCM 语音数据直接传递给 Gemini 驱动执行，拒绝单一脚本文本触发，并实现所有应用全流程深度交互闭环**。

### 测试架构三层分级：
- **Layer 0：安全执行策略与治理规则单元测试** (40 项测试全部通过)
  - 覆盖只读放行、高危按键阻断、Prompt 注入防御、打断令牌流控、单轮预算上限、重复调用抑制、浏览器子进程超时强杀与幽灵数据阻断等。
- **Layer 1：硬件音频与近场 VAD 门控健康检查** (原 Layer 3 提前)
  - 检查麦克风识别、双声道/单声道采集、底噪 RMS、自适应起呼门限与维持门限健康度。
- **Layer 2：真实 PCM 语音端到端全链路闭环评测 (Real Voice-Driven E2E - gemini-3.8-live)**
  - 深度整合所有真实应用深度闭环（Outlook 查收与转发、Word 物理打字保存删除、计算器运算、Ego 浏览器拉到底部提取评论关 space、备忘录新建输入清理、系统运行应用扫描）；
  - 使用高保真真实人声合成（或外部录音 WAV/PCM 文件、麦克风现场录音）打包为标准 16kHz 16-bit 单声道二进制语音流；
  - 语音数据直接传递给 Gemini 3.8 Live 全双工 WebSocket 会话，模型识别语音意图下发工具调用；
  - 底层自动化执行器完成真实深度闭环操作，将执行结果回灌给模型，由模型生成最终中文口语汇报。

### 测试套件常用运行命令：

```bash
# 1. 运行全套完整自动化测试 (Layer 0 ~ Layer 2)
uv run python test_suite.py

# 2. 仅运行指定层级测试
uv run python test_suite.py --layer 0    # 极速安全策略单测 (40项)
uv run python test_suite.py --layer 1    # 硬件麦克风识别与自适应 VAD 门控健康度检查
uv run python test_suite.py --layer 2    # 真实 PCM 语音驱动全链路应用深度闭环评测 (gemini-3.8-live)

# 3. 单独测试特定真实语音闭环用例
uv run python test_suite.py --case outlook    # 语音驱动 Outlook 邮件定位、转发窗口生成与退出
uv run python test_suite.py --case word       # 语音驱动 Word 物理新建、打字、保存与清理
uv run python test_suite.py --case calc       # 语音驱动 计算器物理按钮点击运算与结果验证
uv run python test_suite.py --case browser    # 语音驱动 浏览器新闻浏览、拉底提取评论与关 space
uv run python test_suite.py --case notes      # 语音驱动 备忘录新建、打字、读取与清理
uv run python test_suite.py --case apps       # 语音驱动 系统运行中应用列表扫描

# 4. 自定义真实语音指令驱动实测
uv run python test_suite.py --voice-query "帮我打开Outlook查看第一封邮件"

# 5. 使用外部真实 WAV/PCM 录音文件驱动测试
uv run python test_suite.py --audio-file /path/to/my_voice.wav

# 6. 现场按键从麦克风录音驱动测试
uv run python test_suite.py --record-voice
```

