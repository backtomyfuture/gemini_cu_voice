"""
ego lite (ego-browser) 浏览器自动化集成模块
提供专为语音交互优化的极速页面导航、搜索与正文结构化提取能力
"""
import asyncio
import json
import re
import shutil
import urllib.parse
from typing import Dict, Any, Optional

EGO_BROWSER_BIN = shutil.which("ego-browser") or "/Users/jarod/.local/bin/ego-browser"


async def run_ego_js(js_code: str, timeout: float = 12.0) -> Dict[str, Any]:
    """通过 ego-browser nodejs 执行自动化脚本并解析 JSON 结果"""
    try:
        proc = await asyncio.create_subprocess_exec(
            EGO_BROWSER_BIN, "nodejs",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=js_code.encode("utf-8")),
            timeout=timeout
        )
        combined = stdout.decode("utf-8", errors="replace") + stderr.decode("utf-8", errors="replace")

        # 查找标准 JSON 输出行
        for line in reversed(combined.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}") and '"ok"' in line:
                try:
                    return json.loads(line)
                except Exception:
                    pass

        # 若未找到标准行，返回文本或报错
        if proc.returncode != 0:
            return {"ok": False, "error": combined.strip() or f"进程退出码: {proc.returncode}"}
        return {"ok": True, "raw": combined.strip()[:1000]}
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"浏览器操作超时 ({timeout}s)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def browser_open(url_or_kw: str, max_chars: int = 1200) -> str:
    """
    在 ego lite 浏览器中打开指定网址或搜索关键词，并返回页面标题、URL 和精简正文
    """
    target = url_or_kw.strip()
    # 自动识别是否为 URL，否则转为百度搜索
    if not (target.startswith("http://") or target.startswith("https://")):
        if "." in target and not any(c in target for c in [" ", "，", "？", "！", "、"]):
            target = "https://" + target
        else:
            query_enc = urllib.parse.quote(target)
            target = f"https://www.baidu.com/s?wd={query_enc}"

    code = f"""const task = await taskSpace("voice assistant web");
const page = task.page("p1");
await page.goto("{target}");
await page.waitForLoadState("load");
const title = await page.title();
const currentUrl = await page.url();
const text = await page.evaluate(() => {{
    const ignoreTags = new Set(["SCRIPT", "STYLE", "NOSCRIPT", "NAV", "FOOTER"]);
    function walk(node) {{
        if (!node) return "";
        if (node.nodeType === 3) return node.textContent.trim();
        if (node.nodeType === 1) {{
            if (ignoreTags.has(node.tagName)) return "";
            let res = [];
            for (const child of node.childNodes) {{
                const t = walk(child);
                if (t) res.push(t);
            }}
            return res.join(" ");
        }}
        return "";
    }}
    return walk(document.body);
}});
const cleanText = text.replace(/\\s+/g, " ").slice(0, {max_chars});
console.log(JSON.stringify({{ ok: true, title, url: currentUrl, text: cleanText }}));
"""
    res = await run_ego_js(code, timeout=12.0)
    if res.get("ok"):
        title = res.get("title", "网页")
        url = res.get("url", target)
        text = res.get("text", "")
        return f"【页面标题】: {title}\n【URL】: {url}\n\n【提取正文要点】:\n{text}"
    else:
        return f"打开网页失败: {res.get('error', '未知错误')}"


async def browser_search(query: str, engine: str = "baidu", max_chars: int = 1200) -> str:
    """
    在 ego lite 浏览器中搜索指定内容并提炼核心结果
    """
    q_enc = urllib.parse.quote(query.strip())
    if engine.lower() == "google":
        url = f"https://www.google.com/search?q={q_enc}"
    elif engine.lower() == "bing":
        url = f"https://www.bing.com/search?q={q_enc}"
    else:
        url = f"https://www.baidu.com/s?wd={q_enc}"

    return await browser_open(url, max_chars=max_chars)


async def browser_get_content(max_chars: int = 1500) -> str:
    """
    抓取当前 ego lite 前台页面的正文内容
    """
    code = f"""const task = await taskSpace("voice assistant web");
const page = task.page("p1");
const title = await page.title();
const currentUrl = await page.url();
const text = await page.evaluate(() => {{
    return document.body.innerText.split('\\n')
        .map(s => s.trim())
        .filter(s => s.length > 2)
        .slice(0, 40)
        .join('\\n');
}});
console.log(JSON.stringify({{ ok: true, title, url: currentUrl, text: text.slice(0, {max_chars}) }}));
"""
    res = await run_ego_js(code, timeout=8.0)
    if res.get("ok"):
        return f"【当前页面】: {res.get('title')}\n【URL】: {res.get('url')}\n\n【页面内容】:\n{res.get('text')}"
    return f"获取页面内容失败: {res.get('error', '无法获取')}"


async def browser_click(text_or_selector: str) -> str:
    """
    在当前页面点击指定文字或选择器
    """
    target = text_or_selector.strip()
    code = f"""const task = await taskSpace("voice assistant web");
const page = task.page("p1");
try {{
    if ("{target}".startsWith("#") || "{target}".startsWith(".") || "{target}".startsWith("loc=")) {{
        await page.click("{target}");
    }} else {{
        await page.click('text="{target}"');
    }}
    await page.waitForTimeout(500);
    const title = await page.title();
    console.log(JSON.stringify({{ ok: true, message: "点击成功", title }}));
}} catch (e) {{
    console.log(JSON.stringify({{ ok: false, error: String(e) }}));
}}
"""
    res = await run_ego_js(code, timeout=8.0)
    if res.get("ok"):
        return f"已成功点击 '{target}'，当前页面标题: {res.get('title', '')}"
    return f"点击失败: {res.get('error')}"


async def browser_scroll(direction: str = "down") -> str:
    """
    在当前网页滚动窗口
    """
    delta = 600 if direction.lower() == "down" else -600
    code = f"""const task = await taskSpace("voice assistant web");
const page = task.page("p1");
await page.mouse.wheel(0, {delta});
await page.waitForTimeout(300);
console.log(JSON.stringify({{ ok: true, message: "已滚动页面" }}));
"""
    res = await run_ego_js(code, timeout=5.0)
    if res.get("ok"):
        return f"已向{'下' if delta > 0 else '上'}滚动页面"
    return f"滚动失败: {res.get('error')}"


def get_browser_function_declarations():
    """返回供 Gemini 注册使用的 Ego Lite 浏览器功能声明列表"""
    from google.genai import types

    return [
        types.FunctionDeclaration(
            name="browser_open",
            description="在 Ego Lite 极速浏览器中打开指定的 URL 网址，并抓取提炼网页标题与正文。查网页、看特定网站时使用。",
            parameters={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "要打开的网页完整地址，例如 'https://news.ycombinator.com' 或 'https://github.com'"
                    }
                },
                "required": ["url"]
            }
        ),
        types.FunctionDeclaration(
            name="browser_search",
            description="在 Ego Lite 极速浏览器中联网搜索关键词（支持百度、Google、Bing），提炼核心搜索结果与资讯要点。搜资料、查新闻、检索任何互联网信息时必须优先使用此工具。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，例如 '特斯拉 Roadster 2026', '英伟达最新财报'"
                    },
                    "engine": {
                        "type": "string",
                        "description": "搜索引擎，可选 'baidu', 'google', 'bing'，默认为 'baidu'"
                    }
                },
                "required": ["query"]
            }
        ),
        types.FunctionDeclaration(
            name="browser_get_content",
            description="抓取并提炼当前 Ego Lite 浏览器正在浏览页面的正文内容要点。",
            parameters={
                "type": "object",
                "properties": {}
            }
        ),
        types.FunctionDeclaration(
            name="browser_click",
            description="在 Ego Lite 浏览器当前页面中点击指定文本内容的链接或按钮。",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要点击的链接文字或按钮文字，例如 '下一页', '登录', '新闻'"
                    }
                },
                "required": ["text"]
            }
        ),
        types.FunctionDeclaration(
            name="browser_scroll",
            description="在 Ego Lite 浏览器当前页面中向上或向下滚动屏幕浏览更多内容。",
            parameters={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "滚动方向，'down' (向下滚动) 或 'up' (向上滚动)，默认 'down'"
                    }
                }
            }
        )
    ]

