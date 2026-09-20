# Gemini 3.8 Live + kimi-cu 纯原生实时语音桌面操作管家

这是一个轻量级、超低延迟、全双工的 macOS 桌面实时语音操作管家。

通过 **Google Gemini 3.8 Multimodal Live API**，以全双工双向实时音频流的方式与你交谈，并 100% 通过本机已安装的 **kimi-cu** MCP Server（11 个原生工具）执行桌面应用的识别、置顶全屏、控件点击、按键输入、网页抓取与总结等自动化操作。

---

## 核心特性

- ⚡ **超低延迟全双工语音**：基于 Gemini 3.8 Live API，说话与播报几乎零等待。
- 🗣️ **支持实时高声打断（Barge-in）**：在 AI 播报时，随时开口即可立即打断播报。
- 🎙️ **智能双门限近场语音门控**：
  - 自动过滤远场他人闲聊与环境杂音（敲键盘、关门、脚步声等），绝不误触。
  - 开机 1.0 秒自适应底噪校准，支持通过 `--threshold` 自由调节门限。
- 🖱️ **纯原生联动 kimi-cu**：自动直连本机 `/Applications/KimiCU.app/Contents/MacOS/kimi-cu`，不使用任何系统外挂。
- 🛡️ **严格状态机与看门狗自愈**：彻底解决多步操作虚假就绪、死锁和长连接超时挂起问题。
- 📦 **免配环境**：借助 `uv` 自动管理运行环境与依赖，一键即跑。

---

## 快速运行

### 方式 1：直接运行启动脚本（推荐）
```bash
/Users/jarod/Documents/gemini_cu_voice/run.sh
```

- **自定义防杂音门限**（数值越大越只认贴近大声说，默认 380）：
  ```bash
  /Users/jarod/Documents/gemini_cu_voice/run.sh --threshold 420
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
