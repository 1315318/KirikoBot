from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from flask import Flask, Response, jsonify, render_template, request, send_from_directory

from ai_server import AiServer
from ai_tools import (
    Tarot, Tarot_History, GamingNews,
    WebSearchTool, WeatherTool, StickerTool,
    HitokotoTool, FoodPickerTool, DiceTool, BilibiliTool,
    AtMemberTool, ReminderTool, TimeTool, PoliticalNewsTool,
    BalanceTool, FeatureRequestTool, MusicTool,
    ListRemindersTool, DeleteReminderTool,
)
from balance_service import BalanceService
from ai_tools_list import AiTools
from config import Config
from database_manager import DatabaseManager
from extra_services import HitokotoService, BilibiliTrending
from hot_news import HotNewsScraper
from llbot_client import LLBotClient, MessageBuilder
from msg_package import MsgPackage
from news_crawler import NewsCrawler
from log_stream import sse_handler, setup_sse_logging
from learning_service import LearningService
from music_service import MusicService
from political_news import PoliticalNewsScraper
from profile_service import ProfileService
from robot_server import RobotServer
from scheduler import BotScheduler
from sticker_collector import StickerCollector, STICKER_DIR, STICKER_CATEGORIES
from version_manager import VersionManager
from web_search import WebSearch
from weather_service import WeatherService


def _setup_logging() -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    if root.handlers:
        for x in root.handlers: root.removeHandler(x)
    root.addHandler(h)

_setup_logging()
setup_sse_logging()
logger = logging.getLogger(__name__)

# ── App ─────────────────────────────────────────────────
app = Flask(__name__)

# ── Services ────────────────────────────────────────────
llbot = LLBotClient(Config.ONEBOT_API or "http://llbot:3000", Config.ONEBOT_TOKEN or "")
db = DatabaseManager()
pkg = MsgPackage()
tools_def = AiTools()

tarot = Tarot(db, pkg)
tarot_history = Tarot_History(db, pkg)
news_crawler = NewsCrawler()
gaming_news = GamingNews(news_crawler, pkg)
web_search = WebSearch()
web_search_tool = WebSearchTool(web_search, pkg)
weather_tool = WeatherTool(WeatherService(), pkg)
sticker_tool = StickerTool(pkg)
hitokoto_service = HitokotoService()
hitokoto_tool = HitokotoTool(hitokoto_service, pkg)
food_picker_tool = FoodPickerTool(pkg)
dice_tool = DiceTool(pkg)
bilibili_tool = BilibiliTool(BilibiliTrending(), pkg)
at_member_tool = AtMemberTool(pkg, db, llbot)
reminder_tool = ReminderTool(db, pkg)
list_reminders_tool = ListRemindersTool(db, pkg)
delete_reminder_tool = DeleteReminderTool(db, pkg)
time_tool = TimeTool(pkg)
political_news_scraper = PoliticalNewsScraper()
political_news_tool = PoliticalNewsTool(political_news_scraper, pkg)
balance_service = BalanceService()
balance_tool = BalanceTool(balance_service, pkg)
feature_request_tool = FeatureRequestTool(db, pkg)
music_service = MusicService()
music_tool = MusicTool(music_service, pkg)
hot_news_scraper = HotNewsScraper()

scheduler = BotScheduler(db, llbot, political_news_scraper, news_crawler, hitokoto_service)
scheduler.start()
sticker_collector = StickerCollector(db=db)
profile_service = ProfileService()
learning_service = LearningService()
version_manager = VersionManager(db, llbot)
version_manager.seed_initial_version()

# Dedicated logger for thinking chains — propagates to root (SSE + stdout)
think_log = logging.getLogger("think")

executor = ThreadPoolExecutor(max_workers=12)
sticker_collector.set_executor(executor)
_seeded_groups: set[str] = set()
_start_time = time.time()

# ── Sticker understanding state ─────────────────────────
_sticker_pending: dict[str, float] = {}  # "user_id:group_id" → timestamp
_sticker_pending_lock = threading.Lock()
STICKER_REQUEST_TIMEOUT = 30  # seconds


# Keywords that indicate user wants sticker analysis
STICKER_INTENT_KEYWORDS = [
    "表情包", "这张图", "这图", "表情", "贴纸", "sticker",
    "看看", "看一下", "帮我看看", "图片", "这个图", "看看这个",
]

# ── Tool routing ────────────────────────────────────────
ROUTES = {
    "tarot": tarot.tarot_call, "tarot_history": tarot_history.tarot_history_call,
    "gaming_news": gaming_news.gaming_news_call, "web_search": web_search_tool.web_search_call,
    "weather": weather_tool.weather_call, "sticker": sticker_tool.sticker_call,
    "hitokoto": hitokoto_tool.hitokoto_call, "food_picker": food_picker_tool.food_picker_call,
    "dice": dice_tool.dice_call, "bilibili_trending": bilibili_tool.bilibili_call,
    "at_member": at_member_tool.at_member_call, "set_reminder": reminder_tool.set_reminder_call,
    "list_reminders": list_reminders_tool.list_reminders_call,
    "delete_reminder": delete_reminder_tool.delete_reminder_call,
    "get_current_time": time_tool.get_current_time_call,
    "political_news": political_news_tool.political_news_call,
    "check_balance": balance_tool.balance_call,
    "submit_feature": feature_request_tool.feature_request_call,
    "music_search": music_tool.music_search_call,
}

# ── Multi-turn: tools that should trigger an AI follow-up response ──
FOLLOW_UP_TOOLS = {
    "weather", "food_picker", "dice", "set_reminder",
    "get_current_time", "check_balance", "submit_feature",
    "list_reminders", "delete_reminder",
}

# Self-contained tools format and send their own reply — no AI follow-up needed
SELF_CONTAINED_TOOLS = {
    "tarot", "sticker", "web_search", "at_member",
    "political_news", "gaming_news", "bilibili_trending",
    "hitokoto", "tarot_history", "music_search",
}

# ── History (only recent context, filtered for clarity) ──
MAX_HISTORY = 8  # fewer turns = less noise, more focus on current message

def _load_history(uid: str, gid: str | None) -> list[dict[str, Any]]:
    try:
        rows = db.takeout_chat_history(uid, gid)
    except Exception:
        return []
    history: list[dict[str, Any]] = []
    for role, content, tool_calls, _ in rows:
        if role == "user":
            history.append({"role": "user", "content": content or ""})
        elif role == "assistant":
            # Skip assistant messages that only contain tool calls (no text)
            if content and content.strip():
                history.append({"role": "assistant", "content": content.strip()})
            elif tool_calls:
                # Assistant only called tools, no text — summarize instead of raw JSON
                history.append({"role": "assistant", "content": "[已调用工具处理]"})
    return history[-MAX_HISTORY:]

def _save_turn(uid: str, gid: str | None, user_msg: str, ai_text: str) -> None:
    try:
        db.deposit_chat_history("user", uid, gid, user_msg, "", "")
        if ai_text:
            db.deposit_chat_history("assistant", uid, gid, ai_text, "", "")
    except Exception:
        pass

# ── Group seeding ───────────────────────────────────────
def _seed_group(gid: str) -> None:
    if gid in _seeded_groups:
        return
    _seeded_groups.add(gid)
    try:
        members = llbot.get_group_member_list(gid)
        if members:
            db.seed_group_members(gid, members)
    except Exception:
        pass

# ── Core logic ──────────────────────────────────────────

# Lightweight intent keywords for tool pre-filtering
TOOL_INTENT_MAP: dict[str, list[str]] = {
    "塔罗牌": ["tarot"], "占卜": ["tarot"], "抽牌": ["tarot"], "抽一张": ["tarot"],
    "运势": ["tarot"], "算卦": ["tarot"], "算命": ["tarot"],
    "塔罗历史": ["tarot_history"], "抽牌记录": ["tarot_history"],
    "点歌": ["music_search"], "放歌": ["music_search"], "来首歌": ["music_search"],
    "我想听": ["music_search"], "放一首": ["music_search"], "来首": ["music_search"],
    "歌曲": ["music_search"], "播放": ["music_search"], "音乐": ["music_search"],
    "歌": ["music_search"],  # catch-all: "来首XXX的歌"
    "天气": ["weather"], "气温": ["weather"], "下雨": ["weather"], "温度": ["weather"],
    "新闻": ["political_news", "gaming_news"], "时政": ["political_news"],
    "热搜": ["bilibili_trending"], "B站": ["bilibili_trending"], "bilibili": ["bilibili_trending"],
    "搜索": ["web_search"], "查一下": ["web_search"], "帮我查": ["web_search"],
    "搜索一下": ["web_search"], "查查": ["web_search"],
    "表情包": ["sticker"], "贴纸": ["sticker"],
    "吃什么": ["food_picker"], "推荐吃什么": ["food_picker"], "不知道吃": ["food_picker"],
    "吃啥": ["food_picker"], "吃点什么": ["food_picker"],
    "掷骰子": ["dice"], "roll": ["dice"], "骰子": ["dice"], "随机数": ["dice"],
    "一言": ["hitokoto"], "名言": ["hitokoto"], "语录": ["hitokoto"], "来句": ["hitokoto"],
    "提醒": ["set_reminder", "list_reminders", "delete_reminder"], "叫我": ["set_reminder"],
    "余额": ["check_balance"], "额度": ["check_balance"],
    "建议": ["submit_feature"], "希望能": ["submit_feature"],
    "能不能加": ["submit_feature"], "加一个": ["submit_feature"],
    "@": ["at_member"],
    "游戏新闻": ["gaming_news"], "游戏资讯": ["gaming_news"],
}

# Tools always available even for plain chat (commonly useful, low cost)
FALLBACK_TOOLS = ["sticker"]

def _filter_tools(user_msg: str) -> list[dict[str, Any]]:
    """Pre-filter tools by message content to reduce noise and hallucination risk.
    - Keyword match → matched tools + sticker
    - No match but substantive (>10 chars) → common tools (web_search, music, dice, food, sticker)
    - Trivial/short → sticker only
    Returns filtered tool list."""
    all_tools = tools_def.ai_tools()
    msg_lower = user_msg.lower()

    matched: set[str] = set()
    for keyword, tool_names in TOOL_INTENT_MAP.items():
        if keyword.lower() in msg_lower:
            matched.update(tool_names)

    if matched:
        matched.update(FALLBACK_TOOLS)
        return [t for t in all_tools if t["function"]["name"] in matched]

    # No keyword match — check if message has enough substance to warrant common tools
    if len(user_msg.strip()) > 10:
        matched = {"sticker", "web_search", "music_search", "dice", "food_picker", "hitokoto"}
        return [t for t in all_tools if t["function"]["name"] in matched]

    # Very short / trivial — sticker only
    return [t for t in all_tools if t["function"]["name"] == "sticker"]

def _build_system_prompt(robot: RobotServer) -> str:
    """Build the system prompt with role, tool rules, profiles, and learning context.
    All behavioral instructions live here — the AI treats system messages with highest priority."""
    from datetime import datetime
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    weekday = ["一", "二", "三", "四", "五", "六", "日"][datetime.now().weekday()]

    is_private = robot.msg_type == "private"
    base_role = (Config.PRIVATE_ROLE if is_private else Config.GROUP_ROLE) or ""

    parts: list[str] = [base_role]

    # ── Time context ──
    parts.append(f"当前时间：{now} 周{weekday}")

    # ── Tool usage rules (compact but strict) ──
    parts.append(
        "【工具使用规则】"
        "只根据当前这条消息决定是否调用工具。不要受历史消息影响。"
        "普通聊天/打招呼/感谢/简单问答 → 直接回复，不调用任何工具。"
        "只有当前消息明确要求某功能时才调用对应工具。"
        "不确定时宁可文字回复也不乱调工具。禁止编造任何功能结果。"
    )

    # ── Group-specific rules ──
    if not is_private:
        parts.append("你是群聊机器人，只在群内回复，不要建议私聊。")

    # ── Profile context (system-level, for understanding users) ──
    if not is_private and robot.group_id:
        try:
            profile_text = profile_service.build_context_prompt(db, robot.group_id, robot.user_id)
            if profile_text:
                parts.append(profile_text)
        except Exception:
            pass

    # ── Learning notes (system-level, accumulated behavioral lessons) ──
    try:
        learning_text = learning_service.get_context(db, robot.user_id)
        if learning_text:
            parts.append(learning_text)
    except Exception:
        pass

    return "\n\n".join(parts)

def _context(robot: RobotServer) -> str:
    """Build the user message — clean, focused, just the current interaction."""
    if robot.msg_type == "group":
        return (f"群「{robot.group_name or ''}」中 "
                f"用户 {robot.user_name} 说：{robot.msg}")
    else:
        return f"用户 {robot.user_name} 说：{robot.msg}"

def _log_thinking(user_name: str, reasoning: str) -> None:
    """Log thinking chain to dedicated logger (visible in logs + frontend)."""
    if not reasoning:
        return
    # Truncate very long chains for readability
    preview = reasoning[:800] + "…" if len(reasoning) > 800 else reasoning
    think_log.info("【%s】%s", user_name, preview)



def _process_sticker_analysis(robot: RobotServer, image_url: str) -> None:
    """Analyze a sticker/image via unified vision API, then use DeepSeek for natural reply.

    Flow: Vision API (desc + emotion + category in one call) → DeepSeek AI → reply.
    If vision API is unavailable, falls back to context-based response.
    """
    try:
        vision_desc: str | None = None

        # Step 1: Unified vision call → description + emotion + category
        if Config.VISION_API_URL:
            try:
                vision_data = AiServer.vision_analyze_with_category(image_url)
                if vision_data:
                    vision_desc = vision_data.get("description", "").strip()
                    logger.info(
                        "Vision: %s → cat=%s emo=%s desc=%s",
                        robot.user_name,
                        vision_data.get("category", "?"),
                        vision_data.get("emotion", ""),
                        vision_desc[:40],
                    )
                    # Try to save categorization to matching sticker in DB
                    try:
                        for s in db.get_stickers():
                            fn = s.get("filename", "")
                            if fn and (fn in image_url or image_url.endswith(fn)):
                                db.update_sticker_category(
                                    fn, vision_data.get("category", "未分类"),
                                    vision_desc, vision_data.get("emotion", ""),
                                )
                                break
                    except Exception:
                        pass
            except Exception:
                logger.exception("Vision description failed, falling back to context")

        # Step 2: Feed description (or context) to DeepSeek for natural reply
        if vision_desc:
            user_text = (
                f"用户 {robot.user_name} 发了一个表情包。"
                f"图片内容描述：{vision_desc}"
            )
            if robot.msg.strip():
                user_text += f" 用户同时说：{robot.msg.strip()}"
            system_text = (
                (Config.GROUP_ROLE or "") + "\n"
                "有群友发了一张表情包/图片。上面是图片的描述。"
                "请根据描述用可爱自然的语气对这个表情包做出回应，30字以内。"
                "不要重复描述内容，直接回应即可。"
            )
        else:
            user_text = f"用户 {robot.user_name} 发了一个表情包/图片。"
            if robot.msg.strip():
                user_text += f" 用户同时说：{robot.msg.strip()}"
            system_text = (
                (Config.GROUP_ROLE or "") + "\n"
                "有群友发了一张表情包/图片。你看不到图片内容，"
                "请根据上下文对这张表情包做出可爱的回应，30字以内。"
            )

        ai = AiServer(
            system_text=system_text,
            user_text=user_text,
            history_list=[],
            tools=[],
            model_type="deepseek-v4-flash",
            thinking_type="disabled",
        )
        ai.ai_request()

        reply_text = ai.ai_text.strip() if ai.ai_text else "收到表情包啦～好可爱！(◕‿◕✿)"
        robot.reply(reply_text)

        # Record to chat history
        history_content = "[图片消息]"
        if vision_desc:
            history_content = f"[图片: {vision_desc}]"
        _save_turn(robot.user_id, robot.group_id, history_content, reply_text)

    except Exception:
        logger.exception("Sticker analysis failed for %s", image_url[:60])
        try:
            robot.reply("收到表情包啦～(◕‿◕✿)")
        except Exception:
            pass


def _trigger_profile_update(robot: RobotServer) -> None:
    """Check if user needs profile analysis and submit if so."""
    if robot.msg_type != "group" or not robot.group_id:
        return
    try:
        if profile_service.should_analyze(db, robot.user_id, robot.group_id):
            executor.submit(
                profile_service.analyze_user,
                db, robot.user_id, robot.group_id, robot.user_name,
            )
    except Exception:
        pass


def main_logic(robot: RobotServer) -> None:
    try:
        # ── Skip bot's own messages (echo prevention) ──────
        if str(robot.user_id) == (Config.ROBOT_QQ or ""):
            return

        # Record group message (include image presence in content)
        if robot.msg_type == "group" and robot.group_id:
            _seed_group(robot.group_id)
            msg_content = robot.msg.strip()
            if not msg_content and robot.incoming.has_images:
                msg_content = "[图片消息]"
            if msg_content:
                db.record_group_message(robot.group_id, robot.user_id, robot.user_name, msg_content, robot.user_role or "")

        # ── Sticker understanding flow ────────────────────
        now = time.time()
        pending_key = f"{robot.user_id}:{robot.group_id or 'private'}"
        has_images = robot.incoming.has_images
        first_image_url = robot.incoming.image_urls[0] if robot.incoming.image_urls else ""

        # Clean expired pending requests
        with _sticker_pending_lock:
            expired = [k for k, v in _sticker_pending.items() if now - v > STICKER_REQUEST_TIMEOUT]
            for k in expired:
                del _sticker_pending[k]

        # Check for pending sticker request from this user (2-step flow)
        has_pending = False
        with _sticker_pending_lock:
            if pending_key in _sticker_pending:
                has_pending = True
                del _sticker_pending[pending_key]

        if has_pending:
            if has_images:
                logger.info("🎯 Sticker flow: pending request found, analyzing image from %s", robot.user_name)
                _process_sticker_analysis(robot, first_image_url)
                return
            # User sent text instead of image — clear pending, continue normally

        # ── @bot + image → analyze ──────────────────────
        if robot.at_judgement and robot.msg_type == "group":
            if has_images:
                logger.info("🎯 Sticker flow: direct analysis (@bot+image) from %s", robot.user_name)
                _process_sticker_analysis(robot, first_image_url)
                return

            # No image — sticker intent keywords OR empty message: set pending
            has_sticker_keywords = any(kw in robot.msg for kw in STICKER_INTENT_KEYWORDS)
            no_text = not robot.msg.strip()
            if has_sticker_keywords or no_text:
                with _sticker_pending_lock:
                    _sticker_pending[pending_key] = now
                logger.info("🎯 Sticker flow: pending set for %s, waiting for image", robot.user_name)
                robot.reply("好的，把表情包发给我看看吧～(っ´▽`)っ")
                return

        # ── Private chat with images — always analyze ──
        if has_images and robot.msg_type == "private":
            logger.info("🎯 Sticker flow: private chat image from %s", robot.user_name)
            _process_sticker_analysis(robot, first_image_url)
            return

        # Only respond to @bot or private (after sticker flow)
        if not robot.at_judgement and robot.msg_type != "private":
            return

        # Evaluate previous AI response based on this user message (async)
        executor.submit(
            learning_service.evaluate_and_learn, db, robot.user_id, robot.msg,
        )

        # Trigger profile analysis for group messages (async, non-blocking)
        _trigger_profile_update(robot)

        history = _load_history(robot.user_id, robot.group_id)
        user_text = _context(robot)
        system_prompt = _build_system_prompt(robot)
        is_private = robot.msg_type == "private"

        # Filter tools based on message content — plain chat gets minimal tools
        active_tools = _filter_tools(robot.msg)

        if is_private:
            logger.info("Private chat with %s (%d tools)", robot.user_name, len(active_tools))
        else:
            logger.info("Group chat with %s (%d tools)", robot.user_name, len(active_tools))

        ai = AiServer(system_prompt, user_text, history, active_tools,
                      model_type="deepseek-v4-pro", thinking_type="enabled")
        ai.ai_request()

        # Log thinking chain
        _log_thinking(robot.user_name, ai.reasoning_content)

        tool_calls = ai.ai_message.get("tool_calls") if ai.ai_message else None
        if tool_calls:
            tc_list = tool_calls if isinstance(tool_calls, list) else [tool_calls]
            follow_up_tcs: list[dict[str, Any]] = []

            for tc in tc_list:
                fn = tc.get("function", {}).get("name", "")
                handler = ROUTES.get(fn)
                if not handler:
                    continue
                try:
                    db.record_tool_usage(fn, robot.user_id, robot.group_id)
                except Exception:
                    pass

                if fn in SELF_CONTAINED_TOOLS:
                    handler(robot, ai)
                else:
                    handler(robot, ai)
                    follow_up_tcs.append(tc)
                    tc_id = tc.get("id", "")
                    result_text = getattr(ai, "tool_result_text", "") or ai.user_text or ""
                    ai.add_tool_result(tc_id, result_text)

            if follow_up_tcs:
                ai.follow_up_request(follow_up_tcs)
                _log_thinking(robot.user_name, ai.reasoning_content)
                if ai.ai_text:
                    robot.reply(ai.ai_text)
        elif ai.ai_text:
            robot.reply(ai.ai_text)

        _save_turn(robot.user_id, robot.group_id, robot.msg, ai.ai_text)

        # Record turn for learning (evaluated on next user message)
        tool_name = ""
        if tool_calls:
            names = [tc.get("function", {}).get("name", "") for tc in (tool_calls if isinstance(tool_calls, list) else [tool_calls])]
            tool_name = ",".join(names)
        learning_service.record_turn(robot.user_id, robot.msg, ai.ai_text or "", tool_name)

    except Exception:
        logger.exception("Error for user %s", robot.user_id)
        try: robot.reply("抱歉，处理消息时遇到了问题，请稍后再试~")
        except Exception: pass

# ── HTTP routes ─────────────────────────────────────────
@app.route("/", methods=["GET"])
def dashboard(): return render_template("dashboard.html")

@app.route("/status")
def status():
    # Count stickers
    sticker_count = 0
    try:
        sticker_count = len([f for f in os.listdir(STICKER_DIR) if os.path.isfile(os.path.join(STICKER_DIR, f))])
    except Exception:
        pass
    uptime_sec = int(time.time() - _start_time)
    return jsonify({"ok": True, "model": "deepseek-v4-pro", "thinking": "enabled",
                    "tools": len(tools_def.ai_tools()), "groups": len(_seeded_groups),
                    "uptime": uptime_sec,
                    "scheduler": scheduler._running, "stickers": sticker_count})

@app.route("/stream")
def stream():
    def gen():
        while True:
            entries = sse_handler.read()
            if entries: yield f"data: {json.dumps(entries, ensure_ascii=False)}\n\n"
            else: yield ": keepalive\n\n"
            time.sleep(0.5)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/stats")
def api_stats():
    return jsonify({"totals": db.get_total_stats(),
                    "tools": [{"name": r[0], "count": r[1]} for r in db.get_tool_stats()]})

@app.route("/api/tarot")
def api_tarot(): return jsonify({"records": db.get_all_tarot_history(50)})

@app.route("/api/history")
def api_history(): return jsonify({"records": db.get_all_history(50)})

@app.route("/api/profiles")
def api_profiles(): return jsonify({"profiles": db.get_all_profiles()})

@app.route("/api/learning")
def api_learning():
    rows = db.fetch_data(
        "SELECT id, user_id, note, user_msg, ai_text, tool_name, timestamp FROM learning_log ORDER BY id DESC LIMIT 100"
    )
    return jsonify({"notes": [
        {
            "id": r[0], "user_id": r[1], "note": r[2],
            "user_msg": r[3] or "", "ai_text": r[4] or "",
            "tool_name": r[5] or "", "time": r[6],
        }
        for r in rows
    ]})

@app.route("/api/learning", methods=["POST"])
def api_learning_create():
    data = request.get_json(silent=True) or {}
    user_id = (data.get("user_id") or "dashboard").strip()
    note = (data.get("note") or "").strip()
    if not note:
        return jsonify({"ok": False, "error": "笔记内容不能为空"}), 400
    tool_name = (data.get("tool_name") or "").strip()
    user_msg = (data.get("user_msg") or "").strip()
    ai_text = (data.get("ai_text") or "").strip()
    try:
        db.execute_action(
            "INSERT INTO learning_log (user_id, note, tool_name, user_msg, ai_text) VALUES (?, ?, ?, ?, ?)",
            (user_id, note, tool_name, user_msg, ai_text),
        )
        return jsonify({"ok": True, "msg": "学习笔记已添加"})
    except Exception:
        logger.exception("Failed to create learning note")
        return jsonify({"ok": False, "error": "数据库写入失败"}), 500

@app.route("/api/learning/<int:note_id>", methods=["DELETE"])
def api_learning_delete(note_id: int):
    try:
        db.execute_action("DELETE FROM learning_log WHERE id = ?", (note_id,))
        return jsonify({"ok": True, "deleted": note_id})
    except Exception:
        logger.exception("Failed to delete learning note #%d", note_id)
        return jsonify({"ok": False, "error": "数据库删除失败"}), 500

@app.route("/api/features")
def api_features():
    rows = db.fetch_data(
        "SELECT id, user_name, request_text, category, priority, status, ai_summary, timestamp "
        "FROM feature_requests ORDER BY id DESC LIMIT 100"
    )
    return jsonify({"features": [
        {"id": r[0], "user_name": r[1], "request": r[2], "category": r[3],
         "priority": r[4], "status": r[5], "summary": r[6], "time": r[7]}
        for r in rows
    ]})

@app.route("/api/features/<int:feature_id>", methods=["PATCH"])
def api_features_update(feature_id: int):
    data = request.get_json(silent=True) or {}
    allowed_fields = {"status", "priority", "category"}
    updates = {k: v for k, v in data.items() if k in allowed_fields and v}
    if not updates:
        return jsonify({"ok": False, "error": "No valid fields to update"}), 400
    valid_statuses = {"pending", "done", "rejected"}
    if "status" in updates and updates["status"] not in valid_statuses:
        return jsonify({"ok": False, "error": f"Invalid status. Must be one of: {valid_statuses}"}), 400
    # Check current status before updating (to prevent duplicate changelog entries)
    old_status = ""
    if updates.get("status") == "done":
        old_rows = db.fetch_data(
            "SELECT status FROM feature_requests WHERE id = ?", (feature_id,)
        )
        if old_rows:
            old_status = old_rows[0][0]
    set_clause = ", ".join(f"{k} = ?" for k in updates)
    values = list(updates.values()) + [feature_id]
    try:
        db.execute_action(f"UPDATE feature_requests SET {set_clause} WHERE id = ?", tuple(values))
        # Auto-add changelog entry when a feature is newly marked as done (not re-done)
        if updates.get("status") == "done" and old_status != "done":
            fr_rows = db.fetch_data(
                "SELECT request_text, ai_summary, user_name FROM feature_requests WHERE id = ?",
                (feature_id,),
            )
            if fr_rows:
                fr_request, fr_summary, fr_user = fr_rows[0]
                version_manager.auto_changelog_for_feature(fr_request, fr_summary, fr_user)
        return jsonify({"ok": True, "updated": updates})
    except Exception:
        logger.exception("Failed to update feature #%d", feature_id)
        return jsonify({"ok": False, "error": "Database update failed"}), 500

@app.route("/api/features/<int:feature_id>", methods=["DELETE"])
def api_features_delete(feature_id: int):
    try:
        db.execute_action("DELETE FROM feature_requests WHERE id = ?", (feature_id,))
        return jsonify({"ok": True, "deleted": feature_id})
    except Exception:
        logger.exception("Failed to delete feature #%d", feature_id)
        return jsonify({"ok": False, "error": "Database delete failed"}), 500

@app.route("/api/features", methods=["POST"])
def api_features_create():
    data = request.get_json(silent=True) or {}
    request_text = (data.get("request") or "").strip()
    if not request_text:
        return jsonify({"ok": False, "error": "Request text is required"}), 400
    category = data.get("category", "未分类")
    priority = data.get("priority", "medium")
    user_name = data.get("user_name", "dashboard")
    user_id = data.get("user_id", "admin")
    try:
        db.deposit(
            "feature_requests",
            "(user_id, user_name, group_id, request_text, category, priority, status, ai_summary)",
            "(?, ?, ?, ?, ?, ?, 'pending', ?)",
            (user_id, user_name, None, request_text, category, priority, request_text[:20]),
        )
        return jsonify({"ok": True, "created": {"request": request_text, "category": category, "priority": priority}})
    except Exception:
        logger.exception("Failed to create feature request")
        return jsonify({"ok": False, "error": "Database insert failed"}), 500

# ── Version & Changelog API ──────────────────────────

@app.route("/api/versions/bump", methods=["POST"])
def api_versions_bump():
    """Bump version number and return the new version string (does not create it)."""
    data = request.get_json(silent=True) or {}
    bump_type = data.get("type", "patch")
    if bump_type not in ("major", "minor", "patch"):
        return jsonify({"ok": False, "error": "type must be: major, minor, or patch"}), 400
    try:
        new_version = version_manager.bump_version(bump_type)
        return jsonify({"ok": True, "version": new_version, "bump": bump_type})
    except Exception:
        logger.exception("Version bump failed")
        return jsonify({"ok": False, "error": "版本号递增失败"}), 500

@app.route("/api/versions/current")
def api_versions_current():
    current = version_manager.get_current_version()
    if not current:
        return jsonify({"ok": False, "error": "No versions found"}), 404
    version_id = current["id"]
    changelogs = version_manager.get_changelogs(version_id=version_id)
    current["changelogs"] = changelogs
    return jsonify({"ok": True, "version": current})

@app.route("/api/versions")
def api_versions():
    versions = version_manager.get_all_versions()
    return jsonify({"ok": True, "versions": versions})

@app.route("/api/versions/<int:version_id>")
def api_version_detail(version_id: int):
    detail = version_manager.get_version_detail(version_id)
    if not detail:
        return jsonify({"ok": False, "error": "Version not found"}), 404
    return jsonify({"ok": True, "version": detail})

@app.route("/api/versions", methods=["POST"])
def api_versions_create():
    data = request.get_json(silent=True) or {}
    version = (data.get("version") or "").strip()
    if not version:
        return jsonify({"ok": False, "error": "Version string is required"}), 400
    # Validate version format: X.Y.Z
    import re
    if not re.match(r"^\d+\.\d+\.\d+$", version):
        return jsonify({"ok": False, "error": "Version must be in format X.Y.Z (e.g. 1.0.0)"}), 400
    description = data.get("description", "")
    author = data.get("author", "dashboard")
    notify = data.get("notify", True)
    try:
        result = version_manager.create_version(version, description, author, notify=notify)
        return jsonify({"ok": True, "version": result})
    except Exception:
        logger.exception("Failed to create version %s", version)
        return jsonify({"ok": False, "error": "Database insert failed"}), 500

@app.route("/api/changelog")
def api_changelog():
    version_id = request.args.get("version_id", type=int)
    entry_type = request.args.get("type")
    limit = request.args.get("limit", 100, type=int)
    entries = version_manager.get_changelogs(version_id=version_id, entry_type=entry_type, limit=limit)
    return jsonify({"ok": True, "changelogs": entries})

@app.route("/api/changelog", methods=["POST"])
def api_changelog_create():
    data = request.get_json(silent=True) or {}
    version_id = data.get("version_id", 0)
    entry_type = data.get("entry_type", "feature")
    title = (data.get("title") or "").strip()
    description = data.get("description", "")
    author = data.get("author", "dashboard")
    if not version_id or not title:
        return jsonify({"ok": False, "error": "version_id and title are required"}), 400
    if entry_type not in {"feature", "fix", "improve", "breaking"}:
        return jsonify({"ok": False, "error": "entry_type must be one of: feature, fix, improve, breaking"}), 400
    try:
        entry = version_manager.add_changelog(version_id, entry_type, title, description, author)
        # Notify groups about the new changelog entry
        try:
            executor.submit(version_manager.notify_changelog_entry, entry)
        except Exception:
            pass
        return jsonify({"ok": True, "entry": entry})
    except Exception:
        logger.exception("Failed to create changelog entry")
        return jsonify({"ok": False, "error": "Database insert failed"}), 500

# ── Manual push notification ─────────────────────────

@app.route("/api/changelog/<int:entry_id>/push", methods=["POST"])
def api_changelog_push(entry_id: int):
    """Manually push a changelog entry notification to all groups."""
    rows = db.fetch_data(
        "SELECT id, version_id, entry_type, title, description, author, created_at "
        "FROM changelog WHERE id = ?", (entry_id,)
    )
    if not rows:
        return jsonify({"ok": False, "error": "Changelog entry not found"}), 404
    r = rows[0]
    entry = {
        "id": r[0], "version_id": r[1], "entry_type": r[2],
        "title": r[3], "description": r[4], "author": r[5],
        "created_at": r[6],
    }
    try:
        version_manager.notify_changelog_entry(entry)
        return jsonify({"ok": True, "pushed": entry["title"]})
    except Exception:
        logger.exception("Failed to push changelog entry #%d", entry_id)
        return jsonify({"ok": False, "error": "Push failed"}), 500

@app.route("/api/versions/<int:version_id>/push", methods=["POST"])
def api_version_push(version_id: int):
    """Manually push a version release notification to all groups."""
    detail = version_manager.get_version_detail(version_id)
    if not detail:
        return jsonify({"ok": False, "error": "Version not found"}), 404
    try:
        version_manager.notify_version_release(detail)
        return jsonify({"ok": True, "pushed": detail["version"]})
    except Exception:
        logger.exception("Failed to push version #%d", version_id)
        return jsonify({"ok": False, "error": "Push failed"}), 500

# ── New Feature Digest Push ──────────────────────────

@app.route("/api/digest/push", methods=["POST"])
def api_digest_push():
    """Push a new-feature digest to all active groups. Only pushes current version once."""
    current = version_manager.get_current_version()
    if not current:
        return jsonify({"ok": False, "error": "No version found"}), 404
    version_id = current["id"]
    version_str = current["version"]

    # Check if digest was already sent for this version
    rows = db.fetch_data(
        "SELECT digest_sent FROM app_versions WHERE id = ?", (version_id,)
    )
    if rows and rows[0][0]:
        return jsonify({"ok": False, "error": f"版本 v{version_str} 已推送过速递，无需重复推送"}), 400

    # Get feature-type changelogs from current version ONLY
    features = version_manager.get_changelogs(version_id=version_id, entry_type="feature")
    # Get feature requests completed since this version
    version_created_at = current.get("created_at", "")
    try:
        if version_created_at:
            fr_rows = db.fetch_data(
                "SELECT request_text, ai_summary, user_name FROM feature_requests "
                "WHERE status='done' AND timestamp >= ? ORDER BY id DESC LIMIT 10",
                (version_created_at,)
            )
        else:
            fr_rows = []
        completed_requests = [{"request": r[0], "summary": r[1], "user_name": r[2]} for r in fr_rows]
    except Exception:
        completed_requests = []

    # Build digest message
    lines = [
        "📬 KirikoBot 新功能速递！",
        "",
        f"📦 版本：v{version_str}",
        f"📅 日期：{current.get('release_date', '')}",
        "",
    ]

    if features:
        lines.append("🎉 本次更新内容：")
        for i, f in enumerate(features, 1):
            title = f.get("title", "未知")
            desc = f.get("description", "")
            if desc.startswith("来自 "):
                parts = desc.split("的需求：", 1)
                if len(parts) == 2:
                    desc = parts[1].strip()
            if desc and len(desc) > 80:
                desc = desc[:80] + "…"
            line = f"  {i}. {title}"
            if desc:
                line += f" — {desc}"
            lines.append(line)
        lines.append("")

    if completed_requests:
        lines.append("✅ 近期完成的功能需求：")
        for i, cr in enumerate(completed_requests[:5], 1):
            lines.append(f"  {i}. {cr['summary'] or cr['request'][:20]}（来自 {cr['user_name'] or '群友'}）")
        lines.append("")

    lines.append("感谢大家对 KirikoBot 的支持！(◕‿◕✿)")
    lines.append("有什么想法欢迎 @ 我提建议哦～")

    message = "\n".join(lines)

    # Send to all active groups
    groups = version_manager._get_active_group_ids()
    success = 0
    for gid in groups:
        try:
            from llbot_client import MessageBuilder
            builder = MessageBuilder()
            builder.text(message)
            llbot.send_group_msg(gid, builder.build())
            success += 1
        except Exception:
            logger.exception("Failed to send digest to group %s", gid)

    # Mark digest as sent
    try:
        db.execute_action("UPDATE app_versions SET digest_sent = 1 WHERE id = ?", (version_id,))
    except Exception:
        pass

    logger.info("Feature digest pushed: v%s to %d/%d groups", version_str, success, len(groups))
    return jsonify({"ok": True, "pushed": success, "total_groups": len(groups),
                    "version": version_str, "features_count": len(features)})

@app.route("/api/messages")
def api_messages():
    rows = db.get_recent_group_messages("", 100)
    return jsonify({"records": [{"user_id": r[0], "user_name": r[1], "content": r[2][:100], "time": r[3]} for r in rows]})

@app.route("/api/reminders")
def api_reminders():
    rows = db.fetch_data("SELECT id, user_id, group_id, user_name, content, remind_time, fired, repeat_daily FROM reminders ORDER BY remind_time")
    return jsonify({"reminders": [{"id": r[0], "user_id": r[1], "group_id": r[2], "user_name": r[3],
                                   "content": r[4], "remind_time": r[5], "fired": bool(r[6]),
                                   "repeat_daily": bool(r[7])} for r in rows]})

@app.route("/api/reminders/<int:reminder_id>", methods=["DELETE"])
def api_reminders_delete(reminder_id: int):
    try:
        db.execute_action("DELETE FROM reminders WHERE id = ?", (reminder_id,))
        return jsonify({"ok": True, "deleted": reminder_id})
    except Exception:
        logger.exception("Failed to delete reminder #%d", reminder_id)
        return jsonify({"ok": False, "error": "Database delete failed"}), 500

@app.route("/api/reminders", methods=["POST"])
def api_reminders_create():
    data = request.get_json(silent=True) or {}
    content = (data.get("content") or "").strip()
    remind_time = (data.get("remind_time") or "").strip()
    repeat_daily = int(data.get("repeat_daily", False) or False)
    user_name = data.get("user_name", "dashboard")
    user_id = data.get("user_id", "admin")
    group_id = data.get("group_id") or None
    if not content or not remind_time:
        return jsonify({"ok": False, "error": "content and remind_time are required"}), 400
    from datetime import datetime
    try:
        rt = datetime.strptime(remind_time, "%Y-%m-%d %H:%M:%S")
        if rt <= datetime.now():
            return jsonify({"ok": False, "error": "提醒时间不能是过去的时间"}), 400
    except ValueError:
        return jsonify({"ok": False, "error": "时间格式错误，请使用 YYYY-MM-DD HH:MM:SS"}), 400
    try:
        db.deposit(
            "reminders",
            "(user_id, group_id, user_name, content, remind_time, repeat_daily)",
            "(?, ?, ?, ?, ?, ?)",
            (user_id, group_id, user_name, content, remind_time, repeat_daily),
        )
        return jsonify({"ok": True, "created": {"content": content, "remind_time": remind_time, "repeat_daily": bool(repeat_daily)}})
    except Exception:
        logger.exception("Failed to create reminder")
        return jsonify({"ok": False, "error": "Database insert failed"}), 500

@app.route("/api/balance")
def api_balance():
    result = balance_service.get_balance()
    return jsonify(result)

@app.route("/api/scheduler")
def api_scheduler():
    from datetime import datetime
    now = datetime.now()
    next_morning = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now >= next_morning:
        next_morning = next_morning.replace(day=now.day + 1) if now.month == next_morning.month else now
    next_evening = now.replace(hour=22, minute=0, second=0, microsecond=0)
    if now >= next_evening:
        from datetime import timedelta
        next_evening = next_evening + timedelta(days=1)
    return jsonify({"running": scheduler._running, "check_interval": scheduler.CHECK_INTERVAL,
                    "last_morning": scheduler._last_morning, "last_evening": scheduler._last_evening,
                    "active_groups": scheduler._get_active_groups(),
                    "next_morning": next_morning.strftime("%Y-%m-%d %H:%M"),
                    "next_evening": next_evening.strftime("%Y-%m-%d %H:%M")})

@app.route("/api/scheduler/morning", methods=["POST"])
def api_scheduler_morning():
    try:
        executor.submit(scheduler._morning_greeting)
        return jsonify({"ok": True, "msg": "Morning greeting triggered"})
    except Exception:
        logger.exception("Failed to trigger morning greeting")
        return jsonify({"ok": False, "error": "Failed to trigger"}), 500

@app.route("/api/scheduler/evening", methods=["POST"])
def api_scheduler_evening():
    try:
        executor.submit(scheduler._evening_greeting)
        return jsonify({"ok": True, "msg": "Evening greeting triggered"})
    except Exception:
        logger.exception("Failed to trigger evening greeting")
        return jsonify({"ok": False, "error": "Failed to trigger"}), 500

@app.route("/api/stickers")
def api_stickers():
    category = request.args.get("category", "")
    stickers = []
    try:
        stickers = db.get_stickers(category)
    except Exception:
        pass
    # If no DB records, fall back to file scan
    if not stickers:
        try:
            for f in sorted(os.listdir(STICKER_DIR)):
                fpath = os.path.join(STICKER_DIR, f)
                if os.path.isfile(fpath):
                    size = os.path.getsize(fpath)
                    stickers.append({
                        "filename": f, "file_hash": "", "category": "未分类",
                        "content_desc": "", "emotion": "", "file_size": size,
                        "collected_at": "", "url": f"/stickers/{f}",
                    })
        except Exception:
            pass
    # Add URL to each sticker
    for s in stickers:
        s["url"] = f"/stickers/{s.get('filename', '')}"
    return jsonify({"stickers": stickers, "total": len(stickers)})

@app.route("/api/stickers/categories")
def api_stickers_categories():
    try:
        counts = db.count_stickers_by_category()
        return jsonify({"categories": [{"name": r[0], "count": r[1]} for r in counts]})
    except Exception:
        return jsonify({"categories": []})

@app.route("/api/stickers/<filename>/category", methods=["PATCH"])
def api_stickers_update_category(filename: str):
    data = request.get_json(silent=True) or {}
    category = data.get("category", "").strip()
    if not category:
        return jsonify({"ok": False, "error": "Category is required"}), 400
    if category not in STICKER_CATEGORIES:
        return jsonify({"ok": False, "error": f"无效分类。可选: {', '.join(sorted(STICKER_CATEGORIES))}"}), 400
    content_desc = data.get("content_desc", "")
    emotion = data.get("emotion", "")
    try:
        db.update_sticker_category(filename, category, content_desc, emotion)
        logger.info("Sticker %s category updated to %s", filename, category)
        return jsonify({"ok": True, "updated": {"filename": filename, "category": category}})
    except Exception:
        logger.exception("Failed to update sticker category: %s", filename)
        return jsonify({"ok": False, "error": "数据库更新失败"}), 500

@app.route("/api/stickers/<filename>", methods=["DELETE"])
def api_stickers_delete(filename: str):
    """Delete a sticker file and its DB entry."""
    fpath = os.path.join(STICKER_DIR, filename)
    if not os.path.isfile(fpath):
        return jsonify({"ok": False, "error": "文件不存在"}), 404
    try:
        os.remove(fpath)
        # Remove from DB
        try:
            db.execute_action("DELETE FROM stickers WHERE filename = ?", (filename,))
        except Exception:
            pass
        # Invalidate sticker collector caches
        sticker_collector._hashes = None
        sticker_collector._phashes = None
        logger.info("Sticker deleted: %s", filename)
        return jsonify({"ok": True, "deleted": filename})
    except Exception:
        logger.exception("Failed to delete sticker: %s", filename)
        return jsonify({"ok": False, "error": "文件删除失败"}), 500

@app.route("/api/stickers/orphans/cleanup", methods=["POST"])
def api_stickers_orphans_cleanup():
    """Remove DB entries for stickers whose files no longer exist."""
    try:
        count = db.cleanup_orphan_stickers(STICKER_DIR)
        return jsonify({"ok": True, "cleaned": count})
    except Exception:
        logger.exception("Failed to clean orphan stickers")
        return jsonify({"ok": False, "error": "清理失败"}), 500

# ── Batch sticker organize ────────────────────────────

_sticker_organize_state: dict[str, Any] = {
    "running": False,
    "total": 0,
    "completed": 0,
    "failed": 0,
    "errors": [],
    "started_at": None,
}

@app.route("/api/stickers/organize", methods=["POST"])
def api_stickers_organize():
    """Trigger batch categorization of all uncategorized stickers."""
    if _sticker_organize_state["running"]:
        return jsonify({"ok": False, "error": "批量分类已在运行中"}), 400
    executor.submit(_batch_categorize_stickers)
    return jsonify({"ok": True, "msg": "批量分类已启动"})

@app.route("/api/stickers/organize/progress")
def api_stickers_organize_progress():
    """Return batch categorization progress."""
    return jsonify(_sticker_organize_state)

@app.route("/api/stickers/organize", methods=["DELETE"])
def api_stickers_organize_cancel():
    _sticker_organize_state["running"] = False
    return jsonify({"ok": True, "msg": "分类已取消"})

def _batch_categorize_stickers():
    """Background task: categorize all uncategorized stickers using vision API.

    Calls vision_analyze_with_category() for each uncategorized sticker
    and updates the DB with description, emotion, and category.
    Supports cancellation via _sticker_organize_state["running"].
    """
    _sticker_organize_state["running"] = True
    _sticker_organize_state["completed"] = 0
    _sticker_organize_state["failed"] = 0
    _sticker_organize_state["errors"] = []
    _sticker_organize_state["started_at"] = time.time()

    try:
        # Get all uncategorized stickers from DB
        uncategorized = db.get_uncategorized_stickers()
        _sticker_organize_state["total"] = len(uncategorized)
        logger.info("Batch categorize: %d uncategorized stickers found", len(uncategorized))

        for fname, file_hash in uncategorized:
            if not _sticker_organize_state["running"]:
                break

            fpath = os.path.join(STICKER_DIR, fname)
            if not os.path.isfile(fpath):
                _sticker_organize_state["failed"] += 1
                _sticker_organize_state["errors"].append(f"{fname}: file not found")
                continue

            try:
                vision_data = AiServer.vision_analyze_with_category(fpath)
                if vision_data:
                    db.update_sticker_category(
                        fname,
                        vision_data.get("category", "其他"),
                        vision_data.get("description", ""),
                        vision_data.get("emotion", ""),
                    )
                    _sticker_organize_state["completed"] += 1
                else:
                    # Vision API returned None — leave as uncategorized
                    db.update_sticker_category(fname, "未分类", "", "")
                    _sticker_organize_state["completed"] += 1
            except Exception as e:
                _sticker_organize_state["failed"] += 1
                _sticker_organize_state["errors"].append(f"{fname}: {str(e)[:80]}")

            # Rate limit: 0.5s between vision API calls
            time.sleep(0.5)

            done = _sticker_organize_state["completed"] + _sticker_organize_state["failed"]
            if done % 5 == 0:
                logger.info("Batch categorize: %d/%d (failed: %d)",
                             _sticker_organize_state["completed"],
                             _sticker_organize_state["total"],
                             _sticker_organize_state["failed"])

        duration = int(time.time() - (_sticker_organize_state["started_at"] or time.time()))
        logger.info("Batch categorize finished: %d categorized, %d failed in %ds",
                     _sticker_organize_state["completed"],
                     _sticker_organize_state["failed"], duration)
    finally:
        _sticker_organize_state["running"] = False

# ── Sticker dedup endpoints ──────────────────────────

_sticker_dedup_state: dict[str, Any] = {
    "running": False,
    "scan_result": None,
    "cleanup_result": None,
}

@app.route("/api/stickers/duplicates")
def api_stickers_duplicates():
    """Scan for visually similar duplicate stickers using perceptual hash."""
    force = request.args.get("force", "0") == "1"
    if _sticker_dedup_state["running"] and not force:
        return jsonify({"ok": False, "error": "扫描已在运行中"}), 400

    executor.submit(_scan_duplicates)
    return jsonify({"ok": True, "msg": "重复扫描已启动"})

@app.route("/api/stickers/duplicates/progress")
def api_stickers_duplicates_progress():
    """Return duplicate scan results."""
    return jsonify({
        "running": _sticker_dedup_state["running"],
        "scan_result": _sticker_dedup_state["scan_result"],
        "cleanup_result": _sticker_dedup_state["cleanup_result"],
    })

@app.route("/api/stickers/duplicates/cleanup", methods=["POST"])
def api_stickers_duplicates_cleanup():
    """Remove duplicate stickers, keeping the best quality one from each group."""
    if _sticker_dedup_state["running"]:
        return jsonify({"ok": False, "error": "请等待当前操作完成"}), 400

    dry_run = request.args.get("dry_run", "1") == "1"
    executor.submit(_cleanup_duplicates, dry_run)
    return jsonify({"ok": True, "msg": f"去重清理已启动（{'预览模式' if dry_run else '执行模式'}）"})

def _scan_duplicates():
    """Background task: scan for visually similar stickers."""
    _sticker_dedup_state["running"] = True
    _sticker_dedup_state["scan_result"] = None
    try:
        groups = sticker_collector.find_duplicates()
        result = []
        for group in groups:
            group_info = []
            for f in group:
                group_info.append({
                    "filename": f["filename"],
                    "file_size": f["file_size"],
                    "phash": f.get("phash", ""),
                })
            result.append(group_info)

        total_dups = sum(len(g) - 1 for g in groups)
        waste_bytes = sum(
            sum(f["file_size"] for f in g[1:]) for g in groups
        )
        _sticker_dedup_state["scan_result"] = {
            "total_stickers": len(sticker_collector.hashes),
            "groups": len(groups),
            "duplicate_files": total_dups,
            "waste_bytes": waste_bytes,
            "details": result,
        }
        logger.info("Duplicate scan complete: %d groups, %d duplicates, %d bytes wasted",
                     len(groups), total_dups, waste_bytes)
    except Exception:
        logger.exception("Duplicate scan failed")
        _sticker_dedup_state["scan_result"] = {"error": "扫描失败"}
    finally:
        _sticker_dedup_state["running"] = False

def _cleanup_duplicates(dry_run: bool):
    """Background task: remove duplicate stickers."""
    _sticker_dedup_state["running"] = True
    _sticker_dedup_state["cleanup_result"] = None
    try:
        result = sticker_collector.cleanup_duplicates(dry_run=dry_run)
        _sticker_dedup_state["cleanup_result"] = result
        logger.info(
            "Duplicate cleanup (%s): %d groups, %d removed, %d kept, %d bytes freed",
            "dry_run" if dry_run else "executed",
            result["groups_cleaned"], result["files_removed"],
            result["files_kept"], result["total_waste_bytes"],
        )
    except Exception:
        logger.exception("Duplicate cleanup failed")
        _sticker_dedup_state["cleanup_result"] = {"error": "清理失败"}
    finally:
        _sticker_dedup_state["running"] = False

@app.route("/api/groups")
def api_groups():
    groups = []
    for gid in _seeded_groups:
        try:
            rows = db.fetch_data(
                "SELECT COUNT(*), MAX(timestamp) FROM group_messages WHERE group_id = ?",
                (gid,),
            )
            msg_count = rows[0][0] if rows else 0
            last_active = rows[0][1] if rows and rows[0][1] else ""
        except Exception:
            msg_count = 0
            last_active = ""
        # Get group name from cache or API
        gname = ""
        try:
            info = llbot.get_group_info(gid)
            gname = info.get("group_name", "") if info else ""
        except Exception:
            pass
        groups.append({"group_id": gid, "group_name": gname or gid, "msg_count": msg_count, "last_active": last_active})
    groups.sort(key=lambda g: g["msg_count"], reverse=True)
    return jsonify({"groups": groups, "total": len(groups)})

@app.route("/stickers/<path:filename>")
def serve_sticker(filename: str):
    return send_from_directory(STICKER_DIR, filename)

@app.route("/webhook", methods=["POST"])
@app.route("/", methods=["POST"])
def receive():
    msg_data = request.json
    if not msg_data: return jsonify({"status": "nodata"}), 400

    # ── Only process message events; skip notices (recalls, pokes, etc.) ──
    post_type = msg_data.get("post_type", "message")
    if post_type != "message":
        # Log recall events for debugging but don't process them
        notice_type = msg_data.get("notice_type", "")
        if notice_type:
            logger.info(
                "Ignoring notice event: type=%s user=%s group=%s",
                notice_type, msg_data.get("user_id", ""), msg_data.get("group_id", ""),
            )
        return jsonify({"status": "ignored", "reason": f"post_type={post_type}"}), 200

    if msg_data.get("message_type") == "group":
        executor.submit(sticker_collector.collect, msg_data)
    try:
        robot = RobotServer(msg_data, llbot, Config.ROBOT_QQ or "")
    except Exception:
        return jsonify({"status": "error"}), 400
    executor.submit(main_logic, robot)
    return jsonify({"status": "success"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
