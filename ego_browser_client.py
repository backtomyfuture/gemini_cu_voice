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

    safe_target = json.dumps(target)
    code = f"""const task = await taskSpace("voice assistant web");
const page = task.page("p1");
const targetUrl = {safe_target};
try {{
    await page.goto(targetUrl, {{ waitUntil: "domcontentloaded", timeout: 6500 }});
}} catch (e) {{
    const currentUrl = await page.url();
    const title = await page.title();
    if (!currentUrl || currentUrl === "about:blank" || currentUrl === targetUrl) {{
        console.log(JSON.stringify({{ ok: false, error: "网址无法访问或网络不可达: " + (e.message || String(e)) }}));
        return;
    }}
}}

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
    res = await run_ego_js(code, timeout=9.0)
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


async def browser_get_content(max_chars: int = 1800) -> str:
    """
    抓取当前 ego lite 前台最新页面的正文内容
    """
    code = f"""const task = await taskSpace("voice assistant web");
const tabs = await task.tabs();
const targetTab = tabs[tabs.length - 1];
const page = targetTab && targetTab.label ? task.page(targetTab.label) : task.page("p1");

const title = await page.title();
const currentUrl = await page.url();
const text = await page.evaluate(() => {{
    return document.body.innerText.split('\\n')
        .map(s => s.trim())
        .filter(s => s.length > 2)
        .slice(0, 60)
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
    在当前页面点击指定文字或选择器（支持DOM原生遍历匹配与新标签页自动跟踪）
    """
    target = text_or_selector.strip()
    safe_target = json.dumps(target)
    code = f"""const task = await taskSpace("voice assistant web");
const tabs = await task.tabs();
const activeTab = tabs.find(t => t.active) || tabs[tabs.length - 1];
const page = activeTab && activeTab.label ? task.page(activeTab.label) : task.page("p1");

const raw = {safe_target};
try {{
    let clicked = false;
    // 1. 如果是 css/xpath 选择器
    if (raw.startsWith("#") || raw.startsWith(".") || raw.startsWith("//") || raw.startsWith("a[")) {{
        await page.click(raw, {{ timeout: 3000 }});
        clicked = true;
    }} else {{
        // 2. 原生 DOM 查找与点击，严格匹配，防止误点其他无关链接
        clicked = await page.evaluate((txt) => {{
            const elements = Array.from(document.querySelectorAll("a, button, [role='button'], h1, h2, h3, h4, span, div, p"));
            const cleanTxt = txt.trim().toLowerCase();
            
            // 2.1 精确匹配
            let match = elements.find(el => el.innerText && el.innerText.trim().toLowerCase() === cleanTxt);
            
            // 2.2 包含完整目标词 (要求目标词不少于2个字符)
            if (!match && cleanTxt.length >= 2) {{
                match = elements.find(el => el.innerText && el.innerText.trim().toLowerCase().includes(cleanTxt));
            }}
            
            // 2.3 较长目标词（>10字）前缀匹配（必须至少前8个字严格连续包含）
            if (!match && cleanTxt.length >= 10) {{
                const prefix = cleanTxt.slice(0, 8);
                match = elements.find(el => el.innerText && el.innerText.toLowerCase().includes(prefix));
            }}
            
            if (match) {{
                const anchor = match.closest("a") || match.closest("button") || match;
                anchor.scrollIntoView({{ block: "center" }});
                anchor.click();
                return true;
            }}
            return false;
        }}, raw);

        if (!clicked) {{
            console.log(JSON.stringify({{ ok: false, error: "未在页面中找到包含 '" + raw + "' 的可点击链接或内容。请先调用 browser_get_content 查看页面当前实际显示的文本。" }}));
            return;
        }}
    }}

    await page.waitForTimeout(1000);
    // 检查是否产生了新 Tab（如 target="_blank"）
    const newTabs = await task.tabs();
    const targetTab = newTabs[newTabs.length - 1];
    const targetPage = targetTab && targetTab.label ? task.page(targetTab.label) : page;
    const title = await targetPage.title();
    const url = await targetPage.url();
    console.log(JSON.stringify({{ ok: true, message: "点击成功", title, url }}));
}} catch (e) {{
    console.log(JSON.stringify({{ ok: false, error: String(e) }}));
}}
"""
    res = await run_ego_js(code, timeout=10.0)
    if res.get("ok"):
        return f"已成功点击 '{target}'，当前页面标题: {res.get('title', '')}"
    return f"点击失败: {res.get('error')}"


async def browser_scroll(direction: str = "down") -> str:
    """
    在当前网页滚动窗口（自动作用于最新打开的页面或标签页）
    """
    delta = 800 if direction.lower() == "down" else -800
    code = f"""const task = await taskSpace("voice assistant web");
const tabs = await task.tabs();
const targetTab = tabs[tabs.length - 1];
const page = targetTab && targetTab.label ? task.page(targetTab.label) : task.page("p1");

await page.evaluate((d) => window.scrollBy(0, d), {delta});
await page.waitForTimeout(600);
console.log(JSON.stringify({{ ok: true, message: "已滚动页面" }}));
"""
    res = await run_ego_js(code, timeout=6.0)
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

