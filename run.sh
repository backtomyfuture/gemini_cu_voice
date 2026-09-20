#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

# 自动检测本地代理端口（如 Clash 7890），确保 Gemini WebSocket 连接稳定畅通
if [ -z "$http_proxy" ] && [ -z "$https_proxy" ]; then
    if nc -z -w 1 127.0.0.1 7890 2>/dev/null; then
        export http_proxy="http://127.0.0.1:7890"
        export https_proxy="http://127.0.0.1:7890"
        unset all_proxy
    elif nc -z -w 1 127.0.0.1 7897 2>/dev/null; then
        export http_proxy="http://127.0.0.1:7897"
        export https_proxy="http://127.0.0.1:7897"
        unset all_proxy
    fi
fi

# 优先使用 uv 运行，极速免手动配环境
if command -v uv >/dev/null 2>&1; then
    exec uv run "$DIR/gemini_live_cu.py" "$@"
elif [ -x "$HOME/.local/bin/uv" ]; then
    exec "$HOME/.local/bin/uv" run "$DIR/gemini_live_cu.py" "$@"
else
    # 回退到普通 python3
    if [ ! -d "$DIR/.venv" ]; then
        echo "正在创建虚拟环境并安装依赖..."
        python3 -m venv "$DIR/.venv"
        "$DIR/.venv/bin/pip" install google-genai mcp sounddevice python-dotenv numpy
    fi
    exec "$DIR/.venv/bin/python" "$DIR/gemini_live_cu.py" "$@"
fi
