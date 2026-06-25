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
            "user_msg": user_msg[:500],
            "ai_text": ai_text[:500],
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

        # Build richer evaluation prompt with context
        tool_info = prev['tool'] or '无(直接回复)'
        prompt = (
            f"评估AI表现并生成学习笔记：\n"
            f"- 用户原话：{prev['user_msg']}\n"
            f"- AI调用的工具：{tool_info}\n"
            f"- AI回复内容：{prev['ai_text']}\n"
            f"- 用户随后反馈：{follow_up_msg}\n\n"
            "根据用户反馈判断AI表现。可能的教训类型：\n"
            "- 工具选择错误：用户想要A功能但AI调用了B工具\n"
            "- 遗漏工具：用户明确需要某功能但AI直接回复未调用工具\n"
            "- 回复不当：AI回复偏离用户意图或语气不当\n"
            "- 表现良好：AI正确理解并满足了用户需求\n"
            "用一句话总结（20-50字），格式：'[类型] 具体教训'。只输出教训文本。"
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
                        {"role": "system", "content": "你是一个AI行为评估器，根据用户反馈总结AI表现教训。输出简洁的一句话。"},
                        {"role": "user", "content": prompt},
                    ],
                    "model": "deepseek-v4-flash",
                    "max_tokens": 100,
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

        # Save to database with context
        try:
            db.execute_action(
                "INSERT INTO learning_log (user_id, note, user_msg, ai_text, tool_name) VALUES (?, ?, ?, ?, ?)",
                (user_id, note, prev['user_msg'][:200], prev['ai_text'][:200], prev['tool']),
            )
        except Exception:
            # Fallback for old schema without extra columns
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
