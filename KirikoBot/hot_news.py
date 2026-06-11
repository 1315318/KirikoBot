from __future__ import annotations

import logging
import random
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class HotNewsScraper:
    """Scrapes general hot news from multiple sources."""

    TIMEOUT = 10
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36",
    }
    MAX_ITEMS = 8

    def fetch_all(self) -> list[dict[str, str]]:
        """Fetch hot news from multiple sources, deduplicate, return top items."""
        items: list[dict[str, str]] = []

        # Source 1: Zhihu hot list (JSON API)
        zhihu = self._fetch_zhihu()
        items.extend(zhihu)

        # Source 2: Weibo hot search
        weibo = self._fetch_weibo()
        items.extend(weibo)

        # Source 3: 3DM gaming news (already have this, skip or include)
        gaming = self._fetch_gaming()
        items.extend(gaming)

        # Deduplicate by title
        seen: set[str] = set()
        unique: list[dict[str, str]] = []
        for item in items:
            key = item["title"][:30]
            if key not in seen:
                seen.add(key)
                unique.append(item)
            if len(unique) >= self.MAX_ITEMS:
                break

        return unique

    def _fetch_zhihu(self) -> list[dict[str, str]]:
        try:
            r = requests.get(
                "https://www.zhihu.com/api/v3/feed/topstory/hot-lists/total?limit=10",
                headers=self.HEADERS, timeout=self.TIMEOUT,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            return [
                {"title": d.get("target", {}).get("title", ""), "source": "知乎", "link": ""}
                for d in data[:5] if d.get("target", {}).get("title")
            ]
        except Exception:
            logger.debug("Zhihu hot list fetch failed")
            return []

    def _fetch_weibo(self) -> list[dict[str, str]]:
        try:
            r = requests.get(
                "https://weibo.com/ajax/side/hotSearch",
                headers={**self.HEADERS, "Referer": "https://weibo.com/"},
                timeout=self.TIMEOUT,
            )
            r.raise_for_status()
            data = r.json().get("data", {}).get("realtime", [])
            return [
                {"title": d.get("word", ""), "source": "微博", "link": ""}
                for d in data[:8] if d.get("word")
            ]
        except Exception:
            logger.debug("Weibo hot search fetch failed")
            return []

    def _fetch_gaming(self) -> list[dict[str, str]]:
        """Light gaming news sample."""
        try:
            r = requests.get(
                "https://www.3dmgame.com/news/",
                headers=self.HEADERS, timeout=self.TIMEOUT,
            )
            r.raise_for_status()
            r.encoding = "utf-8"
            soup = BeautifulSoup(r.text, "lxml")
            items = []
            for li in soup.select("li.selectpost")[:3]:
                a = li.select_one("a.bt")
                if a:
                    items.append({"title": a.get_text(strip=True), "source": "游戏", "link": ""})
            return items
        except Exception:
            return []
