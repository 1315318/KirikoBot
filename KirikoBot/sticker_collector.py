from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

import os as _os
_DOCKER_DIR = "/app/stickers"
_LOCAL_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "stickers")
try:
    _os.makedirs(_DOCKER_DIR, exist_ok=True)
    STICKER_DIR = _DOCKER_DIR
except (PermissionError, OSError):
    _os.makedirs(_LOCAL_DIR, exist_ok=True)
    STICKER_DIR = _LOCAL_DIR

MAX_SIZE = 10 * 1024 * 1024  # 10MB
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
# If True (default), only save images with subType=1 (stickers/表情包).
# Set to False to save all images regardless (original behavior).
# NOTE: LLBot may not provide subType in all cases. When subType is missing,
# we default to collecting the image (conservative: collect more, not less).
STICKER_ONLY = True
STICKER_CATEGORIES = {"可爱", "搞笑", "生气", "惊讶", "悲伤", "打招呼", "鼓励", "庆祝", "动物", "动漫", "其他", "未分类"}


class StickerCollector:
    """Save group-shared images to stickers dir. Deduplicate by content hash."""

    def __init__(self, db: Any = None) -> None:
        os.makedirs(STICKER_DIR, exist_ok=True)
        self._hashes: set[str] | None = None  # lazy init
        self._db = db
        self._executor: Any = None

    def set_executor(self, executor: Any) -> None:
        """Inject thread pool for async auto-categorization."""
        self._executor = executor

    def _build_index(self) -> set[str]:
        """Scan all existing stickers and compute MD5 hashes."""
        hashes: set[str] = set()
        if not os.path.isdir(STICKER_DIR):
            return hashes
        for fname in os.listdir(STICKER_DIR):
            fpath = os.path.join(STICKER_DIR, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                h = self._md5_file(fpath)
                if h:
                    hashes.add(h)
            except Exception:
                pass
        logger.debug("Sticker index built: %d hashes", len(hashes))
        return hashes

    @property
    def hashes(self) -> set[str]:
        if self._hashes is None:
            self._hashes = self._build_index()
        return self._hashes

    @staticmethod
    def _md5_file(filepath: str) -> str | None:
        try:
            h = hashlib.md5()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return None

    @staticmethod
    def _md5_data(data: bytes) -> str:
        return hashlib.md5(data).hexdigest()

    def _is_duplicate(self, data: bytes) -> bool:
        """Full-file MD5 check against existing stickers."""
        return self._md5_data(data) in self.hashes

    def collect(self, msg_data: dict[str, Any]) -> tuple[int, list[str]]:
        message = msg_data.get("message")
        if not isinstance(message, list):
            return (0, [])

        saved = 0
        saved_filenames: list[str] = []
        group_id = str(msg_data.get("group_id", "") or "")
        user_id = str(msg_data.get("user_id", "") or "")
        for seg in message:
            if seg.get("type") != "image":
                continue
            data = seg.get("data", {})
            # Only collect stickers (subType=1) when STICKER_ONLY is set.
            # If subType is missing (LLBot may not provide it), default to collecting.
            if STICKER_ONLY:
                sub_type = data.get("subType")
                if sub_type is not None and str(sub_type) not in ("", "1"):
                    logger.debug("Skipping non-sticker image (subType=%s)", sub_type)
                    continue
            image_url = data.get("url", "")
            image_file = data.get("file", "")

            fname: str | None = None
            if image_url:
                fname = self._download(image_url)
            elif image_file and os.path.isfile(image_file):
                fname = self._copy_local(image_file)

            if fname:
                saved += 1
                saved_filenames.append(fname)
                # Record in database if available
                if self._db:
                    fpath = os.path.join(STICKER_DIR, fname)
                    try:
                        file_size = os.path.getsize(fpath)
                        file_hash = self._md5_file(fpath) or ""
                        self._db.insert_sticker(fname, file_hash, file_size, group_id, user_id)
                    except Exception:
                        pass
                # Auto-categorize asynchronously using vision API
                if self._executor and self._db:
                    try:
                        url_for_vision = image_url or os.path.join(STICKER_DIR, fname)
                        self._executor.submit(self._auto_categorize, fname, url_for_vision)
                    except Exception:
                        pass

        return (saved, saved_filenames)

    def _download(self, url: str) -> str | None:
        try:
            ext = os.path.splitext(url.split("?")[0])[1].lower()
            if ext not in VALID_EXTENSIONS:
                ext = ".png"

            # HEAD check for size
            try:
                head = requests.head(url, timeout=5, allow_redirects=True)
                cl = int(head.headers.get("content-length", 0))
                if cl > MAX_SIZE:
                    logger.info("Skipping large: %s (%d)", url[:60], cl)
                    return None
            except Exception:
                pass

            r = requests.get(url, timeout=15, allow_redirects=True, stream=True)
            r.raise_for_status()

            chunks: list[bytes] = []
            total = 0
            for chunk in r.iter_content(chunk_size=8192):
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_SIZE:
                    return None

            data = b"".join(chunks)
            if len(data) < 100:
                return None

            # Deduplicate by full content hash
            if self._is_duplicate(data):
                logger.debug("Skipping duplicate sticker from: %s", url[:60])
                return None

            full_hash = self._md5_data(data)
            fname = f"sticker_{full_hash[:12]}{ext}"
            fpath = os.path.join(STICKER_DIR, fname)

            # Race condition check: if file already exists, skip
            if os.path.exists(fpath):
                self.hashes.add(full_hash)
                return fname  # Already exists — still return name for DB

            with open(fpath, "wb") as f:
                f.write(data)

            self.hashes.add(full_hash)
            logger.info("Collected: %s (%d bytes)", fname, len(data))
            return fname

        except Exception:
            logger.debug("Download failed: %s", url[:60])
            return None

    def _copy_local(self, filepath: str) -> str | None:
        try:
            size = os.path.getsize(filepath)
            if size > MAX_SIZE or size < 100:
                return None
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in VALID_EXTENSIONS:
                return None

            with open(filepath, "rb") as src:
                data = src.read()

            if self._is_duplicate(data):
                logger.debug("Skipping duplicate local: %s", os.path.basename(filepath))
                return None

            full_hash = self._md5_data(data)
            fname = f"sticker_{full_hash[:12]}{ext}"
            fpath = os.path.join(STICKER_DIR, fname)

            if os.path.exists(fpath):
                self.hashes.add(full_hash)
                return fname  # Already exists — still return name for DB

            with open(fpath, "wb") as dst:
                dst.write(data)

            self.hashes.add(full_hash)
            logger.info("Collected local: %s", fname)
            return fname

        except Exception:
            logger.debug("Copy failed: %s", filepath)
            return None

    # ── Auto-categorization ─────────────────────────────

    def _auto_categorize(self, fname: str, image_url_or_path: str) -> None:
        """Mark new sticker as uncategorized in DB. Vision analysis is deferred to
        on-demand @bot interaction (see _process_sticker_analysis in main.py)."""
        if not self._db:
            return
        try:
            self._db.update_sticker_category(fname, "未分类", "", "")
        except Exception:
            logger.debug("Auto-categorize DB update failed for %s", fname)
