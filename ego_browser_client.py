"""
ego lite (ego-browser) 浏览器自动化集成模块
提供专为语音交互优化的极速页面导航、搜索、候选交互清单与正文结构化提取能力
"""
import asyncio
import json
import re
import shutil
import time
import urllib.parse
from typing import Dict, Any, List, Optional

EGO_BROWSER_BIN = shutil.which("ego-browser") or "/Users/jarod/.local/bin/ego-browser"

# 全局活跃标签页与独立 TaskSpace 追踪
_CURRENT_ACTIVE_TAB_LABEL: Optional[str] = None
_CURRENT_SPACE_INDEX: int = 1


def get_current_space_name() -> str:
    """获取当前会话独立的 TaskSpace 名称"""
    return f"voice assistant web {int(time.time() // 3600)}_{_CURRENT_SPACE_INDEX}"


def rotate_task_space():
    """生成下一个全新的独立 TaskSpace，避免复用已结束空间的锁死状态"""
    global _CURRENT_SPACE_INDEX, _CURRENT_ACTIVE_TAB_LABEL
    _CURRENT_SPACE_INDEX += 1
    _CURRENT_ACTIVE_TAB_LABEL = None


def get_active_tab_label() -> Optional[str]:
    """获取当前已追踪的活跃标签页 Label"""
    return _CURRENT_ACTIVE_TAB_LABEL


def set_active_tab_label(label: Optional[str]):
    """显式设置当前活跃标签页 Label"""
    global _CURRENT_ACTIVE_TAB_LABEL
    _CURRENT_ACTIVE_TAB_LABEL = label


def reset_active_tab():
    """重置活跃标签页追踪"""
    global _CURRENT_ACTIVE_TAB_LABEL
    _CURRENT_ACTIVE_TAB_LABEL = None


def build_page_setup_js(preferred_label: Optional[str]) -> str:
    """生成统一的 taskSpace、tabs 检查与目标 page 获取的 JS 前导代码，杜绝冗余、标签解析漂移与 page closed 异常"""
    safe_pref = json.dumps(preferred_label)
    safe_space = json.dumps(get_current_space_name())
    return f"""const preferredLabel = {safe_pref};
const spaceName = {safe_space};
const task = await taskSpace(spaceName);
const tabs = (await task.tabs()) || [];

function isValidLabel(l) {{
    return typeof l === 'string' && /^[a-zA-Z][a-zA-Z0-9_-]*$/.test(l);
}}

let page = null;
let currentActiveLabel = null;
let activeTab = null;

// 1. 如果有指定的活跃 label，且仍存在于当前有效 tabs 中，优先使用
if (preferredLabel && isValidLabel(preferredLabel) && tabs.some(t => t.label === preferredLabel)) {{
    activeTab = tabs.find(t => t.label === preferredLabel);
    try {{
        page = task.page(preferredLabel);
        currentActiveLabel = preferredLabel;
    }} catch (e) {{
        page = null;
    }}
}}

// 2. 从当前存活的 tabs 中挑选活跃或最新标签
if (!page && tabs.length > 0) {{
    activeTab = tabs.find(t => t.active && isValidLabel(t.label)) || tabs.find(t => isValidLabel(t.label));
    if (activeTab && activeTab.label) {{
        try {{
            page = task.page(activeTab.label);
            currentActiveLabel = activeTab.label;
        }} catch (e) {{
            page = null;
        }}
    }}
}}

// 3. 若无可用存活页面，主动创建新页面
if (!page) {{
    page = await task.newPage();
    currentActiveLabel = page.label;
}}
"""


async def run_ego_js(js_code: str, timeout: float = 12.0) -> Dict[str, Any]:
    """通过 ego-browser nodejs 执行自动化脚本并解析 JSON 结果（支持超时强杀子进程与打断协同取消，无僵尸进程残留）"""
    clean_code = js_code.strip()
    if not clean_code.startswith("(async () =>") and not clean_code.startswith("(async()=>"):
        clean_code = f"(async () => {{\n{clean_code}\n}})();"

    proc = None
    try:
        proc = await asyncio.create_subprocess_exec(
            EGO_BROWSER_BIN, "nodejs",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=clean_code.encode("utf-8")),
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

        # 若未找到标准行，返回明确的错误，绝不隐式返回 ok: True
        if proc.returncode != 0:
            return {"ok": False, "error": combined.strip() or f"进程退出码: {proc.returncode}"}
        return {"ok": False, "error": f"浏览器未输出合法的 JSON 响应: {combined.strip()[:500]}"}
    except asyncio.TimeoutError:
        if proc:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        return {"ok": False, "error": f"浏览器操作超时 ({timeout}s)"}
    except asyncio.CancelledError:
        # 关键修复：当协程被外部打断 cancel 时，显式 kill 并 wait 回收子进程，绝不留存后台僵尸进程
        if proc:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
        raise
    except Exception as e:
        if proc:
            try:
                proc.kill()
                await proc.wait()
            except Exception:
                pass
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
    setup_js = build_page_setup_js(_CURRENT_ACTIVE_TAB_LABEL)
    code = f"""{setup_js}
const targetUrl = {safe_target};
try {{
    await page.goto(targetUrl, {{ waitUntil: "domcontentloaded", timeout: 6500 }});
}} catch (e) {{
    console.log(JSON.stringify({{ ok: false, error: "页面导航失败或网络不可达: " + (e.message || String(e)) }}));
    return;
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
console.log(JSON.stringify({{ ok: true, title, url: currentUrl, text: cleanText, activeTabLabel: currentActiveLabel }}));
"""
    res = await run_ego_js(code, timeout=9.0)
    if res.get("ok"):
        if res.get("activeTabLabel"):
            set_active_tab_label(res["activeTabLabel"])
        title = res.get("title", "网页")
        url = res.get("url", target)
        text = res.get("text", "")
        return f"【页面标题】: {title}\n【URL】: {url}\n\n【提取正文要点】:\n{text}"
    else:
        return f"【失败】 打开网页失败: {res.get('error', '未知错误')}"


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
    抓取当前 ego lite 前台最新激活页面的正文内容
    """
    setup_js = build_page_setup_js(_CURRENT_ACTIVE_TAB_LABEL)
    code = f"""{setup_js}
const title = await page.title();
const currentUrl = await page.url();
const text = await page.evaluate(() => {{
    return document.body.innerText.split('\\n')
        .map(s => s.trim())
        .filter(s => s.length > 2)
        .slice(0, 60)
        .join('\\n');
}});
console.log(JSON.stringify({{ ok: true, title, url: currentUrl, text: text.slice(0, {max_chars}), activeTabLabel: currentActiveLabel }}));
"""
    res = await run_ego_js(code, timeout=8.0)
    if res.get("ok"):
        if res.get("activeTabLabel"):
            set_active_tab_label(res["activeTabLabel"])
        return f"【当前页面】: {res.get('title')}\n【URL】: {res.get('url')}\n\n【页面内容】:\n{res.get('text')}"
    return f"【失败】 获取页面内容失败: {res.get('error', '无法获取')}"


async def browser_list_actions(max_items: int = 25) -> str:
    """
    获取当前网页中所有可交互操作元素（链接、按钮、输入项）列表及稳定编号 [#ID]
    用于杜绝重复文案导致的模糊误点击，提供高可靠候选确认能力
    """
    setup_js = build_page_setup_js(_CURRENT_ACTIVE_TAB_LABEL)
    code = f"""{setup_js}
const title = await page.title();
const currentUrl = await page.url();

const items = await page.evaluate((maxCount) => {{
    const selector = "a, button, input[type='button'], input[type='submit'], [role='button'], [role='link']";
    const candidates = Array.from(document.querySelectorAll(selector));
    const results = [];
    let idx = 1;

    for (const el of candidates) {{
        if (results.length >= maxCount) break;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) continue;
        const style = window.getComputedStyle(el);
        if (style.display === "none" || style.visibility === "hidden" || style.opacity === "0") continue;

        let txt = (el.innerText || el.textContent || el.value || el.getAttribute("aria-label") || el.getAttribute("title") || "").trim();
        txt = txt.replace(/\\s+/g, " ");
        if (!txt || txt.length < 2) continue;
        if (txt.length > 70) txt = txt.slice(0, 70) + "...";

        let role = el.tagName.toLowerCase();
        if (el.getAttribute("role")) role = el.getAttribute("role");
        const href = el.getAttribute("href") || "";

        el.setAttribute("data-ego-action-id", String(idx));
        results.push({{
            id: idx,
            role: role,
            text: txt,
            href: href.slice(0, 80)
        }});
        idx++;
    }}
    return results;
}}, {max_items});

console.log(JSON.stringify({{ ok: true, title, url: currentUrl, items, activeTabLabel: currentActiveLabel }}));
"""
    res = await run_ego_js(code, timeout=9.0)
    if not res.get("ok"):
        return f"【失败】 获取页面候选操作失败: {res.get('error', '未知错误')}"
    if res.get("activeTabLabel"):
        set_active_tab_label(res["activeTabLabel"])

    items: List[Dict[str, Any]] = res.get("items", [])
    if not items:
        return f"【当前页面】: {res.get('title', '')}\n【提示】: 当前页面未找到明显的可交互链接或按钮。"

    lines = [f"【页面可交互候选操作列表（共 {len(items)} 项，可直接调用 browser_click(text=\"#ID\") 点击）】:"]
    for it in items:
        role_desc = "链接" if it.get("role") in ["a", "link"] else "按钮"
        href_desc = f" -> {it['href']}" if it.get("href") and not it['href'].startswith("javascript") else ""
        lines.append(f"  [#{it['id']}] [{role_desc}] \"{it['text']}\"{href_desc}")

    return "\n".join(lines)


async def browser_click(text_or_selector: str) -> str:
    """
    在当前页面点击指定文字、选择器或候选编号[#ID]（支持准确 ID 匹配与精准新标签页自动跟踪）
    """
    target = text_or_selector.strip()
    safe_target = json.dumps(target)
    setup_js = build_page_setup_js(_CURRENT_ACTIVE_TAB_LABEL)
    code = f"""{setup_js}
// 记录点击前已存在的标签集合，用于精确判断是否弹出了新标签页，杜绝切到无关历史标签
const beforeTabs = await task.tabs();
const beforeLabels = new Set((beforeTabs || []).map(t => t.label));

const raw = {safe_target};
try {{
    let clicked = false;
    let matchType = "";

    // 1. 编号候选快速点击 (如 "#1", "1", "[#1]", "@1")
    const idMatch = raw.match(/^\\[?#?@?(\\d+)\\]?$/);
    if (idMatch) {{
        const targetId = idMatch[1];
        const findRes = await page.evaluate((tid) => {{
            const el = document.querySelector(`[data-ego-action-id="${{tid}}"]`);
            if (el) {{
                el.scrollIntoView({{ block: "center" }});
                el.click();
                return {{ found: true }};
            }}
            return {{ found: false }};
        }}, targetId);
        if (findRes && findRes.found) {{
            clicked = true;
            matchType = "候选编号ID[#" + targetId + "]";
        }} else {{
            console.log(JSON.stringify({{
                ok: false,
                error: "候选操作编号 [#" + targetId + "] 在当前页面不存在或已失效。请调用 browser_list_actions 重新列举当前页面的可操作元素。"
            }}));
            return;
        }}
    }}

    // 2. CSS/XPath 选择器
    if (!clicked && (raw.startsWith("#") || raw.startsWith(".") || raw.startsWith("//") || raw.startsWith("a["))) {{
        await page.click(raw, {{ timeout: 3000 }});
        clicked = true;
        matchType = "选择器";
    }}

    // 3. 原生 DOM 文本匹配
    if (!clicked) {{
        clicked = await page.evaluate((txt) => {{
            const elements = Array.from(document.querySelectorAll("a, button, [role='button'], h1, h2, h3, h4, span, div, p"));
            const cleanTxt = txt.trim().toLowerCase();

            // 3.1 精确匹配
            let match = elements.find(el => el.innerText && el.innerText.trim().toLowerCase() === cleanTxt);

            // 3.2 包含完整目标词 (不少于2字符)
            if (!match && cleanTxt.length >= 2) {{
                match = elements.find(el => el.innerText && el.innerText.trim().toLowerCase().includes(cleanTxt));
            }}

            // 3.3 前缀连续匹配
            if (!match && cleanTxt.length >= 8) {{
                const prefix = cleanTxt.slice(0, 8);
                match = elements.find(el => el.innerText && el.innerText.toLowerCase().includes(prefix));
            }}

            // 3.4 空格分词联合匹配（针对复合短语如 "IT之家 官方"）
            if (!match && cleanTxt.includes(" ")) {{
                const words = cleanTxt.split(/\\s+/).filter(w => w.length > 0);
                if (words.length > 1) {{
                    match = elements.find(el => {{
                        const t = (el.innerText || "").toLowerCase();
                        return words.every(w => t.includes(w));
                    }});
                    if (!match) {{
                        const containers = Array.from(document.querySelectorAll("a, button, li, tr, [role='listitem'], div.result, div.c-container"));
                        const matchedContainer = containers.find(c => {{
                            const t = (c.innerText || "").toLowerCase();
                            return words.every(w => t.includes(w));
                        }});
                        if (matchedContainer) {{
                            match = matchedContainer.querySelector("a") || matchedContainer.querySelector("button") || matchedContainer;
                        }}
                    }}
                }}
            }}

            if (match) {{
                const anchor = match.closest("a") || match.closest("button") || match;
                anchor.scrollIntoView({{ block: "center" }});
                anchor.click();
                return true;
            }}
            return false;
        }}, raw);
        if (clicked) matchType = "文本匹配";
    }}

    if (!clicked) {{
        console.log(JSON.stringify({{ ok: false, error: "未找到包含 '" + raw + "' 的可点击目标。建议先调用 browser_list_actions 获取精准选项编号。" }}));
        return;
    }}

    await page.waitForTimeout(1000);
    // 关键修复：对比点击前后的标签集合，仅当真正产生新标签页时才切换，杜绝误切到无关已有标签
    const afterTabs = await task.tabs();
    const newlyOpenedTab = (afterTabs || []).find(t => !beforeLabels.has(t.label));
    const targetTab = newlyOpenedTab || activeTab || (afterTabs && afterTabs[0]);
    const targetPage = targetTab && targetTab.label ? task.page(targetTab.label) : page;
    const finalLabel = targetTab && targetTab.label ? targetTab.label : (activeTab && activeTab.label ? activeTab.label : "p1");
    const title = await targetPage.title();
    const url = await targetPage.url();
    console.log(JSON.stringify({{ ok: true, matchType, title, url, activeTabLabel: finalLabel, openedNewTab: Boolean(newlyOpenedTab) }}));
}} catch (e) {{
    console.log(JSON.stringify({{ ok: false, error: String(e) }}));
}}
"""
    res = await run_ego_js(code, timeout=10.0)
    if res.get("ok"):
        if res.get("activeTabLabel"):
            set_active_tab_label(res["activeTabLabel"])
        return f"已成功点击 '{target}'（匹配方式: {res.get('matchType', '文本')}），页面标题: {res.get('title', '')}，当前URL: {res.get('url', '')}"
    return f"【失败】 点击失败: {res.get('error')}"


async def browser_scroll(direction: str = "down") -> str:
    """
    在当前网页滚动窗口（自动作用于最新打开的页面或标签页）
    支持: 'down', 'up', 'bottom' (滚到最底部), 'top' (回到顶部)
    """
    dir_lower = direction.lower()
    if dir_lower in ["bottom", "底部", "最底", "最底部"]:
        scroll_eval = "window.scrollTo(0, document.body.scrollHeight)"
        desc = "拉到最底部"
    elif dir_lower in ["top", "顶部", "最顶", "最顶部"]:
        scroll_eval = "window.scrollTo(0, 0)"
        desc = "回到最顶部"
    elif dir_lower in ["up", "上"]:
        scroll_eval = "window.scrollBy(0, -800)"
        desc = "向上滚动"
    else:
        scroll_eval = "window.scrollBy(0, 800)"
        desc = "向下滚动"

    setup_js = build_page_setup_js(_CURRENT_ACTIVE_TAB_LABEL)
    code = f"""{setup_js}
await page.evaluate(() => {{ {scroll_eval}; }});
await page.waitForTimeout(800);
console.log(JSON.stringify({{ ok: true, message: "已{desc}", activeTabLabel: currentActiveLabel }}));
"""
    res = await run_ego_js(code, timeout=8.0)
    if res.get("ok"):
        if res.get("activeTabLabel"):
            set_active_tab_label(res["activeTabLabel"])
        return f"已成功{desc}页面"
    return f"【失败】 滚动失败: {res.get('error')}"


async def browser_get_comments(max_items: int = 3) -> Dict[str, Any]:
    """
    在当前页面（如新闻、文章、帖子）提取前 N 条用户评论或评论区状态
    """
    setup_js = build_page_setup_js(_CURRENT_ACTIVE_TAB_LABEL)
    code = f"""{setup_js}
const title = await page.title();
const url = await page.url();

const comments = await page.evaluate((maxCount) => {{
    const res = [];
    const selectors = [
        "#ulcommentlist li .comm_content",
        "#ulhotlist li .comm_content",
        ".comm_content",
        ".comment-content",
        ".comment-body",
        ".reply-item",
        ".comment-item",
        ".comm-body",
        "[id^='comm_'] .content",
        ".comm-text",
        "#ulcommentlist li",
        "#ulhotlist li",
        ".comment",
        ".reply-content"
    ];
    const els = Array.from(document.querySelectorAll(selectors.join(", ")));
    for (const el of els) {{
        const t = (el.innerText || el.textContent || "").trim().replace(/\\s+/g, ' ');
        if (t.length >= 2 && !res.includes(t)) {{
            res.push(t);
            if (res.length >= maxCount) break;
        }}
    }}
    if (res.length === 0) {{
        const hintBox = document.querySelector("#pagecomment, #morecomm, #divcommentlist, .comm_list, #post_comment, [id*='comment']");
        if (hintBox) {{
            const hint = (hintBox.innerText || "").trim().replace(/\\s+/g, ' ');
            if (hint && hint.length > 2) {{
                res.push(`[提示] ${{hint.slice(0, 150)}}`);
            }}
        }}
    }}
    return res;
}}, {max_items});

console.log(JSON.stringify({{
    ok: true,
    title,
    url,
    comments,
    count: comments.length,
    activeTabLabel: currentActiveLabel
}}));
"""
    res = await run_ego_js(code, timeout=9.0)
    if res.get("ok"):
        if res.get("activeTabLabel"):
            set_active_tab_label(res["activeTabLabel"])
        return res
    return {"ok": False, "error": res.get("error", "提取评论失败"), "comments": []}


async def browser_close(close_window: bool = False) -> str:
    """
    关闭当前语音助手专属的独立 Ego TaskSpace 中的所有标签页并轮转空间，不退出用户整个浏览器
    """
    safe_space = json.dumps(get_current_space_name())
    code = f"""
const task = await taskSpace({safe_space});
try {{
    const tabs = (await task.tabs()) || [];
    for (const t of tabs) {{
        if (t.label) {{
            try {{
                const p = task.page(t.label);
                await p.close();
            }} catch(err) {{}}
        }}
    }}
    console.log(JSON.stringify({{ ok: true, message: "已成功关闭独立 TaskSpace 页面" }}));
}} catch (e) {{
    console.log(JSON.stringify({{ ok: false, error: String(e) }}));
}}
"""
    await run_ego_js(code, timeout=6.0)
    rotate_task_space()
    return "已成功关闭 Ego Lite 独立任务空间（TaskSpace），保留浏览器正常运行"


def get_browser_function_declarations(
    behavior: Optional[str] = None,
    behavior_map: Optional[Dict[str, str]] = None
):
    """返回供 Gemini 注册使用的 Ego Lite 浏览器功能声明列表，支持按模型指定全局 behavior 或按工具名称指定 behavior_map"""
    from google.genai import types

    def _get_kw(name: str):
        b = (behavior_map.get(name) if behavior_map else None) or behavior
        return {"behavior": b} if b else {}

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
            },
            **_get_kw("browser_open"),
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
            },
            **_get_kw("browser_search"),
        ),
        types.FunctionDeclaration(
            name="browser_get_content",
            description="抓取并提炼当前 Ego Lite 浏览器正在浏览页面的正文内容要点。",
            parameters={
                "type": "object",
                "properties": {}
            },
            **_get_kw("browser_get_content"),
        ),
        types.FunctionDeclaration(
            name="browser_list_actions",
            description="获取当前网页中所有可交互操作元素（链接、按钮）列表及对应编号 [#ID]。当页面有多个相似项或需精确点击时，先调用此工具列出候选编号，再用 browser_click 点击。",
            parameters={
                "type": "object",
                "properties": {
                    "max_items": {
                        "type": "integer",
                        "description": "最多提取的操作项数量，默认 25"
                    }
                }
            },
            **_get_kw("browser_list_actions"),
        ),
        types.FunctionDeclaration(
            name="browser_click",
            description="在 Ego Lite 浏览器当前页面中点击指定链接或按钮。支持传入候选编号（如 '#1' 或 '1'）或直接传入文字标题。",
            parameters={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "要点击的候选编号（如 '#1'）或链接文字、按钮文字"
                    }
                },
                "required": ["text"]
            },
            **_get_kw("browser_click"),
        ),
        types.FunctionDeclaration(
            name="browser_scroll",
            description="在 Ego Lite 浏览器当前页面中向上或向下滚动屏幕浏览更多内容，支持滚动到底部（'bottom'）或顶部（'top'）。",
            parameters={
                "type": "object",
                "properties": {
                    "direction": {
                        "type": "string",
                        "description": "滚动方向，'down' (向下滚动), 'up' (向上滚动), 'bottom' (滚动到底部), 'top' (回到顶部)，默认 'down'"
                    }
                }
            },
            **_get_kw("browser_scroll"),
        ),
        types.FunctionDeclaration(
            name="browser_close",
            description="关闭当前浏览的网页标签页或彻底关闭退出 Ego Lite 浏览器窗口。",
            parameters={
                "type": "object",
                "properties": {
                    "close_window": {
                        "type": "boolean",
                        "description": "是否彻底退出浏览器应用窗口，默认为 true"
                    }
                }
            },
            **_get_kw("browser_close"),
        )
    ]
