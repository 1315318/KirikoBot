#!/usr/bin/env python3
"""Bootstrap affection scores for all profiled users based on existing chat history.

Analyzes each user's: message count, keyword sentiment, tool usage, learning feedback.
Computes initial affection_score and writes to user_affection table.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from typing import Any

DB_PATH = "/home/bosak/Documents/ClaudeCode_Projects/KirikoBot/KirikoBot/robot.db"

# ── Same keyword patterns as affection_service.py ──────────
POSITIVE_PATTERNS: list[tuple[str, float]] = [
    ("最喜欢", 3), ("爱了", 3), ("好喜欢你", 3), ("真棒", 3),
    ("厉害", 2.5), ("好强", 2.5), ("牛", 2), ("太强了", 2.5),
    ("谢谢", 1.5), ("感谢", 1.5), ("多谢", 1.5),
    ("可爱", 2), ("好萌", 2.5), ("贴心", 2.5),
    ("好用", 1.5), ("方便", 1), ("不错", 1),
    ("哈哈", 0.5), ("笑死", 1), ("好有趣", 1.5),
    ("好棒", 2), ("太好了", 2), ("完美", 2),
    ("好评", 1.5), ("真香", 2),
]

NEGATIVE_PATTERNS: list[tuple[str, float]] = [
    ("垃圾", -2.5), ("废物", -3), ("没用", -2.5), ("真没用", -3),
    ("滚", -3), ("闭嘴", -2.5), ("别说了", -2), ("烦死了", -2.5),
    ("笨", -1.5), ("蠢", -2), ("傻逼", -3), ("SB", -3),
    ("不好用", -2), ("什么鬼", -1.5), ("乱说", -2),
    ("无语", -1.5), ("失望", -2), ("差评", -2),
    ("别@我", -2), ("别叫我", -2),
]

SCORE_MIN = 0.0
SCORE_MAX = 100.0
SCORE_DEFAULT = 50.0

RELATIONSHIP_LEVELS = [
    (80, "挚友", "🌟"),
    (60, "亲密", "💕"),
    (40, "友好", "😊"),
    (20, "普通", "👋"),
    (0,  "冷淡", "❄️"),
]


def get_relationship(score: float) -> tuple[str, str]:
    for threshold, label, emoji in RELATIONSHIP_LEVELS:
        if score >= threshold:
            return label, emoji
    return "冷淡", "❄️"


def score_text(text: str) -> float:
    """Score a single message for keyword sentiment."""
    text_lower = text.lower()
    best_pos = 0.0
    best_neg = 0.0
    for pattern, value in POSITIVE_PATTERNS:
        if pattern.lower() in text_lower and value > best_pos:
            best_pos = value
    for pattern, value in NEGATIVE_PATTERNS:
        if pattern.lower() in text_lower and abs(value) > abs(best_neg):
            best_neg = value
    return best_pos + best_neg


def bootstrap_user(
    db: sqlite3.Connection,
    user_id: str,
    group_id: str,
    user_name: str,
) -> dict[str, Any]:
    """Compute initial affection score for one user from their history."""

    # ── 1. Messages ───────────────────────────────────────
    rows = db.execute(
        "SELECT content, timestamp FROM group_messages "
        "WHERE user_id = ? AND group_id = ? ORDER BY id",
        (user_id, group_id),
    ).fetchall()

    msg_count = len(rows)
    if msg_count == 0:
        return None  # no data to score

    # Get last interaction time
    last_ts = rows[-1][1] if rows else None

    # Compute total keyword sentiment (sample up to 500 messages)
    total_sentiment = 0.0
    pos_hits = 0
    neg_hits = 0
    for content, _ in rows[-500:]:
        s = score_text(content or "")
        if s > 0:
            pos_hits += 1
            total_sentiment += s
        elif s < 0:
            neg_hits += 1
            total_sentiment += s

    # Cap sentiment contribution
    sentiment_bonus = max(-10.0, min(20.0, total_sentiment))

    # ── 2. Tool usage ─────────────────────────────────────
    tool_count = db.execute(
        "SELECT COUNT(*) FROM tool_usage WHERE user_id = ? AND group_id = ?",
        (user_id, group_id),
    ).fetchone()[0]
    tool_bonus = min(tool_count * 0.5, 10.0)

    # ── 3. Learning feedback ──────────────────────────────
    learning_rows = db.execute(
        "SELECT note FROM learning_log WHERE user_id = ?", (user_id,)
    ).fetchall()
    learning_bonus = 0.0
    for (note,) in learning_rows:
        if not note:
            continue
        if "表现良好" in note:
            learning_bonus += 2.0
        elif "回复不当" in note or "工具选择错误" in note:
            learning_bonus -= 1.0
        elif "遗漏工具" in note:
            learning_bonus -= 0.5
    learning_bonus = max(-5.0, min(10.0, learning_bonus))

    # ── 4. Message frequency bonus ────────────────────────
    # Active users get up to +15 based on message count
    activity_bonus = min(msg_count / 20.0, 15.0)

    # ── 5. Compute final score ────────────────────────────
    score = SCORE_DEFAULT + activity_bonus + sentiment_bonus + tool_bonus + learning_bonus
    score = max(SCORE_MIN, min(SCORE_MAX, score))
    relationship, emoji = get_relationship(score)

    return {
        "user_id": user_id,
        "group_id": group_id,
        "user_name": user_name,
        "affection_score": round(score, 1),
        "interaction_count": msg_count,
        "positive_count": pos_hits,
        "negative_count": neg_hits,
        "last_interaction": last_ts,
        "relationship": relationship,
        "emoji": emoji,
        "activity_bonus": round(activity_bonus, 1),
        "sentiment_bonus": round(sentiment_bonus, 1),
        "tool_bonus": round(tool_bonus, 1),
        "learning_bonus": round(learning_bonus, 1),
    }


def main() -> None:
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row

    # Get all unique users from group_messages
    users = db.execute(
        "SELECT DISTINCT user_id, group_id, user_name FROM user_profiles"
    ).fetchall()

    # Also get users with messages but no profile yet
    msg_users = db.execute(
        "SELECT DISTINCT user_id, group_id, user_name FROM group_messages "
        "WHERE group_id IS NOT NULL AND group_id != '' GROUP BY user_id, group_id"
    ).fetchall()

    # Merge unique users
    all_users: dict[tuple[str, str], tuple[str, str]] = {}
    for row in users + msg_users:
        key = (row["user_id"], row["group_id"])
        if key not in all_users:
            all_users[key] = (row["user_id"], row["group_id"], row["user_name"])

    print(f"Found {len(all_users)} unique users across all groups\n")

    results = []
    for (uid, gid), (_, _, uname) in all_users.items():
        result = bootstrap_user(db, uid, gid, uname)
        if result is None:
            continue
        results.append(result)

    # Sort by score descending
    results.sort(key=lambda r: r["affection_score"], reverse=True)

    # ── Display ────────────────────────────────────────────
    print(f"{'用户':<16} {'分数':>6} {'关系':<6} {'消息':>5} {'活跃':>6} {'情感':>6} {'工具':>5} {'学习':>5}")
    print("-" * 72)
    for r in results:
        print(
            f"{r['user_name'][:14]:<16} "
            f"{r['affection_score']:>5.0f} "
            f"{r['emoji']}{r['relationship']:<4} "
            f"{r['interaction_count']:>5} "
            f"{r['activity_bonus']:>+5.0f} "
            f"{r['sentiment_bonus']:>+5.0f} "
            f"{r['tool_bonus']:>+4.0f} "
            f"{r['learning_bonus']:>+4.0f}"
        )

    print()
    print(f"Score distribution: ", end="")
    ranges = [("挚友", 80), ("亲密", 60), ("友好", 40), ("普通", 20)]
    for label, threshold in ranges:
        count = sum(1 for r in results if r["affection_score"] >= threshold)
        print(f"{label}: {count}  ", end="")
    print()

    # ── Write to database ──────────────────────────────────
    print("\n写入 user_affection 表...")
    inserted = 0
    updated = 0
    for r in results:
        # Check if record exists
        existing = db.execute(
            "SELECT id FROM user_affection WHERE user_id = ? AND group_id = ?",
            (r["user_id"], r["group_id"]),
        ).fetchone()

        if existing:
            db.execute(
                "UPDATE user_affection SET "
                "affection_score = ?, interaction_count = ?, positive_count = ?, "
                "negative_count = ?, last_interaction = ?, relationship = ?, "
                "user_name = ?, updated_at = datetime('now','localtime') "
                "WHERE user_id = ? AND group_id = ?",
                (
                    r["affection_score"], r["interaction_count"],
                    r["positive_count"], r["negative_count"],
                    r["last_interaction"], r["relationship"],
                    r["user_name"], r["user_id"], r["group_id"],
                ),
            )
            updated += 1
        else:
            db.execute(
                "INSERT INTO user_affection "
                "(user_id, group_id, user_name, affection_score, interaction_count, "
                "positive_count, negative_count, last_interaction, relationship) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    r["user_id"], r["group_id"], r["user_name"],
                    r["affection_score"], r["interaction_count"],
                    r["positive_count"], r["negative_count"],
                    r["last_interaction"], r["relationship"],
                ),
            )
            inserted += 1

    db.commit()
    print(f"  ✅ 新增 {inserted} 条，更新 {updated} 条")
    print(f"  📊 总计 {len(results)} 位用户的好感度已初始化")

    db.close()


if __name__ == "__main__":
    main()
