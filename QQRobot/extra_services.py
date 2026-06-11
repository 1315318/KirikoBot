from __future__ import annotations

import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)

# ── Hitokoto ──────────────────────────────────────────────

class HitokotoService:
    """Fetches random inspirational quotes from hitokoto.cn (free API)."""

    URL = "https://v1.hitokoto.cn/"
    TIMEOUT = 8

    def get_quote(self) -> dict[str, str] | None:
        try:
            r = requests.get(self.URL, timeout=self.TIMEOUT)
            r.raise_for_status()
            data = r.json()
            return {
                "text": data.get("hitokoto", ""),
                "source": data.get("from", "") or "未知",
                "author": data.get("from_who", "") or "",
            }
        except Exception:
            logger.exception("Failed to fetch hitokoto")
            return None


# ── Bilibili Trending ─────────────────────────────────────

class BilibiliTrending:
    """Scrapes Bilibili hot search (no API key)."""

    URL = "https://api.bilibili.com/x/web-interface/wbi/search/square?limit=10&platform=web"
    TIMEOUT = 10
    MAX_ITEMS = 10

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36",
        "Referer": "https://www.bilibili.com/",
    }

    def get_trending(self) -> list[dict[str, str]]:
        try:
            r = requests.get(
                self.URL,
                headers=self.HEADERS,
                timeout=self.TIMEOUT,
            )
            r.raise_for_status()
        except Exception:
            logger.exception("Bilibili trending request failed")
            return []

        try:
            data = r.json()
            items = []
            trending_list = (
                data.get("data", {}).get("trending", {}).get("list", [])
                or data.get("data", {}).get("archives", [])
            )
            for item in trending_list[:self.MAX_ITEMS]:
                keyword = item.get("keyword", "") or item.get("title", "")
                if not keyword:
                    continue
                show_name = item.get("show_name", "") or item.get("desc", "") or ""
                items.append({"keyword": keyword, "desc": show_name})
            return items
        except Exception:
            logger.exception("Failed to parse Bilibili trending")
            return []

    def get_hot_videos(self) -> list[dict[str, str]]:
        """Fallback: scrape popular videos from Bilibili ranking."""
        try:
            r = requests.get(
                "https://api.bilibili.com/x/web-interface/ranking/v2?rid=0&type=all",
                headers=self.HEADERS,
                timeout=self.TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            items = []
            for v in data.get("data", {}).get("list", [])[:self.MAX_ITEMS]:
                items.append({
                    "keyword": v.get("title", ""),
                    "desc": f"UP主: {v.get('owner', {}).get('name', '?')} | 播放: {v.get('stat', {}).get('view', 0)}",
                })
            return items
        except Exception:
            logger.exception("Bilibili ranking request failed")
            return []
