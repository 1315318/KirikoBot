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

# ── History (no tool records stored → no 400 errors) ────
MAX_HISTORY = 16

def _load_history(uid: str, gid: str | None) -> list[dict[str, Any]]:
    try:
        rows = db.takeout_chat_history(uid, gid)
    except Exception:
        return []
    history: list[dict[str, Any]] = []
    for role, content, _, _ in rows:
        if role == "user":
            history.append({"role": "user", "content": content or ""})
        elif role == "assistant" and content:
            history.append({"role": "assistant", "content": content or ""})
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
def _context(robot: RobotServer) -> str:
    from datetime import datetime
    now = datetime.now().strftime("%Y年%m月%d日 %H:%M")
    weekday = ["一", "二", "三", "四", "五", "六", "日"][datetime.now().weekday()]
    time_ctx = f"当前时间：{now} 周{weekday}"

    base = (f"群：{robot.group_name}({robot.group_id}) "
            f"用户：{robot.user_name}({robot.user_id}) "
            f"消息：{robot.msg}") if robot.msg_type == "group" else \
           f"用户：{robot.user_name}({robot.user_id}) 消息：{robot.msg}"

    # Inject group profiles for group chats
    profile_text = ""
    if robot.msg_type == "group" and robot.group_id:
        try:
            profile_text = profile_service.build_context_prompt(db, robot.group_id, robot.user_id)
        except Exception:
            pass

    # Inject learning notes
    learning_text = ""
    try:
        learning_text = learning_service.get_context(db, robot.user_id)
    except Exception:
        pass

    parts = [base, time_ctx]
    if robot.msg_type == "group":
        parts.append("注意：你是群聊机器人，永远不要建议或尝试发送私聊消息。始终在群内回复。")
    if profile_text:
        parts.append(profile_text)
    if learning_text:
        parts.append(learning_text)
    return "\n".join(parts)

def _log_thinking(user_name: str, reasoning: str) -> None:
    """Log thinking chain to dedicated logger (visible in logs + frontend)."""
    if not reasoning:
        return
    # Truncate very long chains for readability
    preview = reasoning[:800] + "…" if len(reasoning) > 800 else reasoning
    think_log.info("【%s】%s", user_name, preview)


def _process_sticker_analysis(robot: RobotServer, image_url: str) -> None:
    """Analyze a sticker image via vision API and reply with natural language."""
    try:
        # Step 1: Get vision analysis
        import json as _json
        analysis_prompt = _json.dumps({
            "task": "用户发了一个表情包/图片，分析它",
            "output": {
                "category": "分类(可爱/搞笑/生气/惊讶/悲伤/打招呼/鼓励/庆祝/动物/动漫/其他)",
                "emotion": "传达的情绪",
                "description": "15字内描述图片内容"
            }
        }, ensure_ascii=False)
        analysis_result = AiServer.vision_analyze(image_url, analysis_prompt, response_format="json")

        try:
            analysis = _json.loads(analysis_result)
        except (_json.JSONDecodeError, ValueError):
            analysis = {"category": "其他", "emotion": "未知", "description": "无法分析"}

        desc = analysis.get("description", analysis.get("category", "未知"))
        emotion = analysis.get("emotion", "")

        # Step 2: Feed analysis result to AI for natural reply
        ai = AiServer(
            system_text=(Config.GROUP_ROLE or "") + "\n用户发了一张图片。根据分析结果，用可爱自然的语气回复，20字以内。",
            user_text=f"用户发了一个表情包。分析结果：类别={analysis.get('category','')}, 情绪={emotion}, 描述={desc}。请简短回复。",
            history_list=[],
            tools=[],
            model_type="deepseek-v4-flash",
            thinking_type="disabled",
        )
        ai.ai_request()

        reply_text = ai.ai_text.strip() if ai.ai_text else f"收到一张{desc}的图片～{'感觉' + emotion if emotion else ''} ✨"
        robot.reply(reply_text)

        # Record to chat history so AI can reference it later
        _save_turn(robot.user_id, robot.group_id, f"[图片] {desc}", reply_text)

    except Exception:
        logger.exception("Sticker analysis failed for %s", image_url[:60])
        try:
            robot.reply("唔…看不太清楚这张图呢，再发一次试试？(｡•́︿•̀｡)")
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
                _process_sticker_analysis(robot, first_image_url)
                return
            # User sent text instead of image — clear pending, continue normally

        # Check if @bot message is a sticker analysis request
        if robot.at_judgement and robot.msg_type == "group":
            is_sticker_request = any(kw in robot.msg for kw in STICKER_INTENT_KEYWORDS)
            if is_sticker_request:
                if has_images:
                    _process_sticker_analysis(robot, first_image_url)
                    return
                else:
                    # Set pending — wait for user to send the sticker
                    with _sticker_pending_lock:
                        _sticker_pending[pending_key] = now
                    robot.reply("好的，把表情包发给我看看吧～(っ´▽`)っ")
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
        is_private = robot.msg_type == "private"

        # ── Private/Group: both use v4-pro with thinking ──
        if is_private:
            system_prompt = Config.PRIVATE_ROLE or ""
            logger.info("Private chat with %s", robot.user_name)
        else:
            system_prompt = Config.GROUP_ROLE or ""

        ai = AiServer(system_prompt, user_text, history, tools_def.ai_tools(),
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
        "SELECT user_id, note, timestamp FROM learning_log ORDER BY id DESC LIMIT 100"
    )
    return jsonify({"notes": [{"user_id": r[0], "note": r[1], "time": r[2]} for r in rows]})

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
    """Push a new-feature digest to all active groups. Summarizes features from the current version."""
    current = version_manager.get_current_version()
    if not current:
        return jsonify({"ok": False, "error": "No version found"}), 404
    version_id = current["id"]
    version_str = current["version"]

    # Get all feature-type changelogs from current version
    features = version_manager.get_changelogs(version_id=version_id, entry_type="feature")
    # Also get recently completed feature requests
    try:
        fr_rows = db.fetch_data(
            "SELECT request_text, ai_summary, user_name FROM feature_requests "
            "WHERE status='done' ORDER BY id DESC LIMIT 10"
        )
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
            # Clean description — extract just the feature request part
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
            summary = cr["summary"] or cr["request"][:20]
            user = cr["user_name"] or "群友"
            lines.append(f"  {i}. {summary}（来自 {user}）")
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
    valid_categories = {"可爱", "搞笑", "生气", "惊讶", "悲伤", "打招呼", "鼓励", "庆祝", "动物", "动漫", "其他", "未分类"}
    if category not in valid_categories:
        return jsonify({"ok": False, "error": f"无效分类。可选: {', '.join(sorted(valid_categories))}"}), 400
    content_desc = data.get("content_desc", "")
    emotion = data.get("emotion", "")
    try:
        db.update_sticker_category(filename, category, content_desc, emotion)
        logger.info("Sticker %s category updated to %s", filename, category)
        return jsonify({"ok": True, "updated": {"filename": filename, "category": category}})
    except Exception:
        logger.exception("Failed to update sticker category: %s", filename)
        return jsonify({"ok": False, "error": "数据库更新失败"}), 500

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
    """Background task: categorize all existing stickers via vision API."""
    import json as _json
    _sticker_organize_state["running"] = True
    _sticker_organize_state["completed"] = 0
    _sticker_organize_state["failed"] = 0
    _sticker_organize_state["errors"] = []
    _sticker_organize_state["started_at"] = time.time()

    try:
        # Get all sticker files
        files = [f for f in os.listdir(STICKER_DIR)
                 if os.path.isfile(os.path.join(STICKER_DIR, f))
                 and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp"))]

        # Filter: skip already categorized
        uncategorized = []
        for fname in files:
            rows = db.fetch_data(
                "SELECT category FROM stickers WHERE filename=? AND category NOT IN ('未分类', '')",
                (fname,),
            )
            if not rows:
                uncategorized.append(fname)

        _sticker_organize_state["total"] = len(uncategorized)
        logger.info("Sticker organize: %d total, %d uncategorized", len(files), len(uncategorized))

        for i, fname in enumerate(uncategorized):
            if not _sticker_organize_state["running"]:
                break

            fpath = os.path.join(STICKER_DIR, fname)

            # Ensure DB entry exists
            existing = db.fetch_data("SELECT filename FROM stickers WHERE filename=?", (fname,))
            if not existing:
                from sticker_collector import StickerCollector
                file_hash = StickerCollector._md5_file(fpath) or ""
                file_size = os.path.getsize(fpath)
                db.insert_sticker(fname, file_hash, file_size)

            # Categorize via vision API
            try:
                analysis_prompt = _json.dumps({
                    "task": "分析这个表情包/图片",
                    "output": {
                        "category": "分类(可爱/搞笑/生气/惊讶/悲伤/打招呼/鼓励/庆祝/动物/动漫/其他)",
                        "emotion": "传达的情绪(5字内)",
                        "description": "10字内描述图片内容"
                    }
                }, ensure_ascii=False)
                result = AiServer.vision_analyze(fpath, analysis_prompt, response_format="json")
                try:
                    data = _json.loads(result)
                    category = data.get("category", "其他")
                    if category not in STICKER_CATEGORIES:
                        category = "其他"
                    desc = data.get("description", "")
                    emotion = data.get("emotion", "")
                except (_json.JSONDecodeError, ValueError):
                    category, desc, emotion = "其他", "", ""
                db.update_sticker_category(fname, category, desc, emotion)
                _sticker_organize_state["completed"] += 1
            except Exception as e:
                _sticker_organize_state["completed"] += 1
                _sticker_organize_state["failed"] += 1
                _sticker_organize_state["errors"].append(f"{fname}: {str(e)[:80]}")

            # Rate limit to avoid API throttling
            time.sleep(0.5)

            if (i + 1) % 20 == 0:
                logger.info("Sticker organize: %d/%d completed (failed %d)",
                            i + 1, len(uncategorized), _sticker_organize_state["failed"])

        duration = int(time.time() - (_sticker_organize_state["started_at"] or time.time()))
        logger.info("Sticker organize finished: %d/%d in %ds (failed %d)",
                     _sticker_organize_state["completed"],
                     _sticker_organize_state["total"], duration,
                     _sticker_organize_state["failed"])
    finally:
        _sticker_organize_state["running"] = False

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
