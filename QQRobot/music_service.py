from __future__ import annotations

import logging
import re
from typing import Any

import requests

logger = logging.getLogger(__name__)


class MusicService:
    """Search songs via Netease Cloud Music API. Returns song info for OneBot music share cards.

    Primary delivery: OneBot 'music' type share card (native QQ music share UI).
    Audio download is attempted but may fail due to Netease anti-hotlinking.
    """

    SEARCH_API = "https://music.163.com/api/search/get"
    TIMEOUT = 15

    # Headers to mimic a browser (required by Netease)
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://music.163.com/",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }

    def search(self, keyword: str, limit: int = 5) -> list[dict[str, Any]]:
        """Search songs by keyword via Netease API. Returns list of song info dicts."""
        # Build params with encoded keyword
        params = {
            "s": keyword,
            "type": 1,      # 1 = song
            "limit": limit,
            "offset": 0,
        }

        try:
            r = requests.get(
                self.SEARCH_API,
                params=params,
                timeout=self.TIMEOUT,
                headers=self.HEADERS,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            logger.exception("Music search API failed for keyword '%s'", keyword)
            return []

        songs = self._parse_results(data)
        logger.info("Music search for '%s': found %d results", keyword, len(songs))
        return songs

    def _parse_results(self, data: Any) -> list[dict[str, Any]]:
        """Parse Netease search response into normalized song dicts."""
        results: list[dict[str, Any]] = []

        try:
            raw_songs = data.get("result", {}).get("songs", [])
        except Exception:
            return results

        for item in raw_songs:
            if not isinstance(item, dict):
                continue

            song_id = item.get("id", 0)

            # Artists
            artists = []
            for a in item.get("artists", []) or item.get("ar", []):
                if isinstance(a, dict):
                    artists.append(a.get("name", ""))

            # Album
            album = item.get("album", {}) or item.get("al", {}) or {}
            album_name = album.get("name", "") if isinstance(album, dict) else ""

            # Cover image
            cover = ""
            if isinstance(album, dict):
                pic_id = album.get("picId", 0)
                if pic_id:
                    cover = f"https://p2.music.126.net/{pic_id}.jpg"

            # Duration
            duration_ms = item.get("duration", 0)
            duration_sec = duration_ms // 1000 if duration_ms else 0

            # Audio URL — try to construct from Netease (may redirect to HTML)
            audio_url = f"https://music.163.com/song/media/outer/url?id={song_id}.mp3"

            results.append({
                "id": song_id,
                "name": item.get("name", "未知歌曲"),
                "artist": " / ".join(artists) if artists else "未知歌手",
                "artists": artists,
                "album": album_name,
                "cover": cover,
                "audio_url": audio_url,
                "duration": duration_sec,
                # OneBot music share card uses this
                "music_type": "163",  # Netease Cloud Music
            })

        return results

    def search_best(self, keyword: str) -> dict[str, Any] | None:
        """Search and return the best matching song."""
        songs = self.search(keyword, limit=3)
        if not songs:
            return None

        # Prefer results with known artists
        for song in songs:
            if song.get("artist") and song["artist"] != "未知歌手":
                return song

        return songs[0]

    def _extract_song_id(self, text: str) -> str | None:
        """Try to extract a Netease song ID from a URL or text."""
        # Match patterns like: id=123456, song?id=123456, /song/123456
        patterns = [
            r"id[=/](\d+)",
            r"/song/(\d+)",
            r"/(\d{5,12})",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m and len(m.group(1)) >= 6:
                return m.group(1)
        return None

    def download_audio(self, audio_url: str, song_id: int) -> str | None:
        """Attempt to download audio file. Returns local path or None.

        Netease anti-hotlinking often blocks this — it's a best-effort attempt.
        """
        import os
        import tempfile

        try:
            r = requests.get(
                audio_url,
                timeout=30,
                headers=self.HEADERS,
                stream=True,
                allow_redirects=True,
            )
            r.raise_for_status()

            # Check if we got actual audio or HTML
            content_type = r.headers.get("Content-Type", "")
            if "text/html" in content_type or "text/plain" in content_type:
                # Likely got a redirect page or error page instead of audio
                # Read a small chunk to confirm
                chunk = r.raw.read(512)
                if chunk.startswith(b"<") or chunk.startswith(b"<!DOCTYPE"):
                    logger.info("Audio URL returned HTML (anti-hotlinking), skipping download")
                    return None
                # Reset and re-download
                r = requests.get(audio_url, timeout=30, headers=self.HEADERS, stream=True)
                r.raise_for_status()

            # Save to temp file
            fd, fpath = tempfile.mkstemp(suffix=".mp3", prefix=f"music_{song_id}_")
            with os.fdopen(fd, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            size_kb = os.path.getsize(fpath) / 1024
            if size_kb < 10:
                # Too small to be real audio
                os.remove(fpath)
                logger.info("Downloaded file too small (%.1f KB), likely not real audio", size_kb)
                return None

            logger.info("Downloaded audio: %s (%.1f KB)", fpath, size_kb)
            return fpath
        except Exception:
            logger.exception("Failed to download audio for song_id=%s", song_id)
            return None
