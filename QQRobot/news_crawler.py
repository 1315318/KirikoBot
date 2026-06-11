from __future__ import annotations

import logging
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class NewsCrawler:
    """Fetches trending gaming news from 3DM."""

    BASE_URL = "https://www.3dmgame.com/news/"
    CACHE_TTL = 300  # seconds
    MAX_ITEMS = 10
    REQUEST_TIMEOUT = 15

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    def __init__(self) -> None:
        self._cache: list[dict[str, str]] | None = None
        self._cache_time: float = 0.0

    def fetch_gaming_news(self) -> list[dict[str, str]]:
        # Return cached data if still fresh
        if self._cache is not None and (time.time() - self._cache_time) < self.CACHE_TTL:
            logger.info("Returning cached gaming news (%d items)", len(self._cache))
            return self._cache

        try:
            response = requests.get(
                self.BASE_URL,
                headers=self.HEADERS,
                timeout=self.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            response.encoding = "utf-8"
        except requests.exceptions.Timeout:
            logger.error("3DM news request timed out")
            return self._fallback()
        except requests.exceptions.ConnectionError:
            logger.exception("3DM news connection error")
            return self._fallback()
        except requests.exceptions.HTTPError:
            logger.exception("3DM news HTTP error")
            return self._fallback()
        except Exception:
            logger.exception("Unexpected error fetching 3DM news")
            return self._fallback()

        try:
            news_items = self._parse(response.text)
        except Exception:
            logger.exception("Failed to parse 3DM news HTML")
            return self._fallback()

        if not news_items:
            logger.warning("No news items extracted from 3DM")
            return self._fallback()

        self._cache = news_items
        self._cache_time = time.time()
        logger.info("Fetched %d gaming news items from 3DM", len(news_items))
        return news_items

    def _fallback(self) -> list[dict[str, str]]:
        """Return cached data if available, otherwise empty list."""
        if self._cache is not None:
            logger.info("Returning stale cache as fallback")
            return self._cache
        return []

    def _parse(self, html: str) -> list[dict[str, str]]:
        soup = BeautifulSoup(html, "lxml")
        items: list[dict[str, str]] = []
        seen_titles: set[str] = set()

        for li in soup.select("li.selectpost"):
            try:
                title_a = li.select_one("a.bt")
                if not title_a:
                    continue
                title = title_a.get_text(strip=True)
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                link = title_a.get("href", "")
                if link and not link.startswith("http"):
                    link = f"https://www.3dmgame.com{link}" if link.startswith("/") else f"https://www.3dmgame.com/{link}"

                category_a = li.select_one("div.bq a.a")
                category = category_a.get_text(strip=True) if category_a else "综合"

                time_span = li.select_one("div.bq span.time")
                news_time = time_span.get_text(strip=True) if time_span else ""

                desc_div = li.select_one("div.miaoshu")
                summary = desc_div.get_text(strip=True) if desc_div else ""

                items.append({
                    "title": title,
                    "link": link,
                    "category": category,
                    "time": news_time,
                    "summary": summary,
                })

                if len(items) >= self.MAX_ITEMS:
                    break
            except Exception:
                logger.warning("Skipping malformed news item")
                continue

        return items
