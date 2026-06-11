from __future__ import annotations

import json
import logging
from typing import Any

import requests

from config import Config

logger = logging.getLogger(__name__)


class LearningService:
    """Lightweight self-improvement: evaluates AI responses based on user reactions.

    After each user message, the previous AI response is evaluated. One-line notes
    are accumulated and injected into future conversations to improve behavior.
    """

    MAX_NOTES = 12          # max learning notes per user
    MIN_MSG_LENGTH = 4      # ignore very short follow-ups (stickers, "ok", etc.)

    def __init__(self) -> None:
        self._pending: dict[str, dict[str, str]] = {}  # user_id -> {user_msg, ai_text, tool_name}

    def record_turn(self, user_id: str, user_msg: str, ai_text: str, tool_name: str) -> None:
        """Cache the current turn for evaluation on the next user message."""
        self._pending[user_id] = {
            "user_msg": user_msg[:200],
            "ai_text": ai_text[:200],
            "tool": tool_name,
        }

    def evaluate_and_learn(
        self, db: Any, user_id: str, follow_up_msg: str,
    ) -> str | None:
        """Evaluate the previous turn based on the user's follow-up message.
        Returns a learning note string, or None if no evaluation is needed."""
        prev = self._pending.pop(user_id, None)
        if not prev:
            return None

        # Skip if follow-up is too short (likely just "ok", "thanks", sticker, etc.)
        if len(follow_up_msg.strip()) < self.MIN_MSG_LENGTH:
            return None

        # Build evaluation prompt
        prompt = (
            f"评估AI表现。用户原话：'{prev['user_msg']}'。"
            f"AI调用了：{prev['tool'] or '无(直接回复)'}。"
            f"AI回复了：'{prev['ai_text']}'。"
            f"用户随后说：'{follow_up_msg}'。"
            "用一句话总结教训（如'用户要新闻时应调用political_news而不是发表情包'），"
            "不超过40字。只输出教训文本，不要其他内容。"
        )

        try:
            r = requests.post(
                Config.DEEPSEEK_API,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {Config.DEEPSEEK_TOKEN}",
                },
                json={
                    "messages": [
                        {"role": "system", "content": "你是一个AI行为评估器，用一句话总结AI哪里做错了或做对了。"},
                        {"role": "user", "content": prompt},
                    ],
                    "model": "deepseek-v4-flash",
                    "max_tokens": 80,
                    "temperature": 0,
                },
                timeout=15,
            )
            r.raise_for_status()
            note = r.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            logger.info("Learning evaluation skipped (API unavailable)")
            return None

        if not note or len(note) < 3:
            return None

        # Save to database
        try:
            db.execute_action(
                "INSERT INTO learning_log (user_id, note) VALUES (?, ?)",
                (user_id, note),
            )
        except Exception:
            logger.debug("Failed to save learning note")
            return None

        logger.info("Learned [%s]: %s", user_id, note)
        return note

    def get_context(self, db: Any, user_id: str) -> str:
        """Get accumulated learning notes for injection into system context."""
        try:
            rows = db.fetch_data(
                "SELECT note FROM learning_log WHERE user_id=? ORDER BY id DESC LIMIT ?",
                (user_id, self.MAX_NOTES),
            )
        except Exception:
            return ""

        if not rows:
            return ""

        notes = [r[0] for r in rows if r[0]]
        if not notes:
            return ""

        return "【学习笔记】过往互动中总结的改进点：\n" + "\n".join(f"  - {n}" for n in notes)
