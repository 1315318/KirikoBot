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
STICKER_ONLY = True


class StickerCollector:
    """Save group-shared images to stickers dir. Deduplicate by content hash."""

    def __init__(self) -> None:
        os.makedirs(STICKER_DIR, exist_ok=True)
        self._hashes: set[str] | None = None  # lazy init

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

    def collect(self, msg_data: dict[str, Any]) -> int:
        message = msg_data.get("message")
        if not isinstance(message, list):
            return 0

        saved = 0
        for seg in message:
            if seg.get("type") != "image":
                continue
            data = seg.get("data", {})
            # Only collect stickers (subType=1), skip regular photos (subType=0/missing)
            if STICKER_ONLY:
                sub_type = data.get("subType")
                if sub_type not in (1, "1"):
                    logger.debug("Skipping non-sticker image (subType=%s)", sub_type)
                    continue
            image_url = data.get("url", "")
            image_file = data.get("file", "")

            if image_url and self._download(image_url):
                saved += 1
            elif image_file and os.path.isfile(image_file) and self._copy_local(image_file):
                saved += 1

        return saved

    def _download(self, url: str) -> bool:
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
                    return False
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
                    return False

            data = b"".join(chunks)
            if len(data) < 100:
                return False

            # Deduplicate by full content hash
            if self._is_duplicate(data):
                logger.debug("Skipping duplicate sticker from: %s", url[:60])
                return False

            full_hash = self._md5_data(data)
            fname = f"sticker_{full_hash[:12]}{ext}"
            fpath = os.path.join(STICKER_DIR, fname)

            # Race condition check: if file already exists, skip
            if os.path.exists(fpath):
                self.hashes.add(full_hash)
                return False

            with open(fpath, "wb") as f:
                f.write(data)

            self.hashes.add(full_hash)
            logger.info("Collected: %s (%d bytes)", fname, len(data))
            return True

        except Exception:
            logger.debug("Download failed: %s", url[:60])
            return False

    def _copy_local(self, filepath: str) -> bool:
        try:
            size = os.path.getsize(filepath)
            if size > MAX_SIZE or size < 100:
                return False
            ext = os.path.splitext(filepath)[1].lower()
            if ext not in VALID_EXTENSIONS:
                return False

            with open(filepath, "rb") as src:
                data = src.read()

            if self._is_duplicate(data):
                logger.debug("Skipping duplicate local: %s", os.path.basename(filepath))
                return False

            full_hash = self._md5_data(data)
            fname = f"sticker_{full_hash[:12]}{ext}"
            fpath = os.path.join(STICKER_DIR, fname)

            if os.path.exists(fpath):
                self.hashes.add(full_hash)
                return False

            with open(fpath, "wb") as dst:
                dst.write(data)

            self.hashes.add(full_hash)
            logger.info("Collected local: %s", fname)
            return True

        except Exception:
            logger.debug("Copy failed: %s", filepath)
            return False
