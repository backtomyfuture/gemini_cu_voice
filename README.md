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

### 测试架构四层分级：
- **Layer 0：安全执行策略与治理规则单元测试** (20 项测试全部通过)
  - 覆盖只读放行、高危按键阻断、Prompt 注入防御、打断令牌流控、单轮预算上限、重复调用抑制、浏览器子进程超时强杀与幽灵数据阻断等。
- **Layer 1：真实应用深度闭环基座测试 (Deep Closed-Loop Integration)**
  - **Outlook 邮件闭环**：打开 Outlook -> 查看到第一封邮件 -> 点开第一封邮件 -> 提取主题、发件人及正文摘要并完整反馈。
  - **Word 文档闭环**：打开 Word -> 新建空白文档 -> 键入文字 -> 保存到登录用户的下载文件夹 (`~/Downloads/*.docx`) -> 验证文件存在且大小有效 -> 关闭 Word -> 从下载文件夹彻底删除该文件 -> 验证完全清理。
  - **计算器闭环**：打开计算器 -> 动态识别并点击无障碍按钮运算 `8 × 9` -> 读取界面实际显示结果 `72` -> 退出关闭应用。
  - **Ego 浏览器闭环**：打开资讯网站 -> 提取候选编号清单 `[#1]` -> 精准点击第一篇新闻 -> 详情页抓取完整正文并反馈。
  - **备忘录闭环**：打开备忘录 -> 新建笔记并录入内容 -> 读取验证 -> 彻底清理删除该测试笔记并退出。
  - **系统应用管理**：扫描全部运行中应用并置顶聚焦。
- **Layer 2：真实 PCM 语音端到端全链路闭环评测 (Real Voice-Driven E2E)**
  - 使用 macOS 高品质真实人声合成（或外部录音 WAV/PCM 文件、麦克风现场录音）打包为标准 16kHz 16-bit 单声道二进制语音流；
  - 语音数据直接传递给 Gemini 智能体，模型通过听音识别用户意图并下发工具动作；
  - 底层自动化执行器完成真实深度闭环操作，并将执行结果回灌给模型；
  - 模型根据实际内容生成最终中文口语汇报，实现“真实声音输入 -> 动作闭环 -> 结果口语反馈”的全链路贯通。
- **Layer 3：硬件音频与近场 VAD 门控健康检查**
  - 检查麦克风识别、双声道/单声道采集、底噪 RMS、自适应起呼门限与维持门限健康度。

### 测试套件常用运行命令：

```bash
# 1. 运行全套完整自动化测试 (Layer 0 ~ Layer 3)
uv run python test_suite.py

# 2. 仅运行指定层级测试
uv run python test_suite.py --layer 0    # 极速安全策略单测 (20项)
uv run python test_suite.py --layer 1    # 真实应用深度闭环基座测试
uv run python test_suite.py --layer 2    # 真实 PCM 语音驱动端到端全链路评测
uv run python test_suite.py --layer 3    # 硬件麦克风与自适应 VAD 健康度检查

# 3. 单独测试特定深度闭环用例 (秒级执行)
uv run python test_suite.py --case outlook    # 单测 Outlook 邮件查收、点开与内容读取
uv run python test_suite.py --case word       # 单测 Word 键入、保存下载文件夹与删除清理
uv run python test_suite.py --case calc       # 单测 计算器按钮点击运算与结果读取
uv run python test_suite.py --case browser    # 单测 浏览器候选点击与正文下钻抓取
uv run python test_suite.py --case notes      # 单测 备忘录新建、读取与清理

# 4. 自定义真实语音指令驱动实测
uv run python test_suite.py --voice-query "帮我打开Outlook查看第一封邮件"

# 5. 使用外部真实 WAV/PCM 录音文件驱动测试
uv run python test_suite.py --audio-file /path/to/my_voice.wav

# 6. 现场按键从麦克风录音驱动测试
uv run python test_suite.py --record-voice
```

