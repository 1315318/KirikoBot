from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests
from requests.adapters import HTTPAdapter, Retry

logger = logging.getLogger(__name__)


# ── Data models ──────────────────────────────────────────

@dataclass
class IncomingMessage:
    """Parsed incoming OneBot message."""
    raw: dict[str, Any]
    msg_type: str  # "group" or "private"
    user_id: str
    user_name: str
    group_id: str | None = None
    group_name: str | None = None
    user_role: str | None = None
    user_level: str | None = None
    user_title: str | None = None
    text: str = ""
    is_at_bot: bool = False
    message_id: int | None = None

    @classmethod
    def from_onebot(cls, data: dict[str, Any], bot_qq: str) -> IncomingMessage:
        msg_type = data.get("message_type", "private")
        user_id = str(data.get("user_id", ""))
        group_id = data.get("group_id")
        if group_id is not None:
            group_id = str(group_id)
        sender = data.get("sender") or {}

        message_raw = data.get("message")
        if not isinstance(message_raw, list):
            message_raw = []

        text = cls._extract_text(message_raw)
        is_at = any(
            seg.get("type") == "at"
            and str(seg.get("data", {}).get("qq", "")) == bot_qq
            for seg in message_raw
        ) if msg_type == "group" else True  # always respond in private

        return cls(
            raw=data,
            msg_type=msg_type,
            user_id=user_id,
            user_name=sender.get("nickname", "unknown"),
            group_id=group_id,
            group_name=data.get("group_name"),
            user_role=sender.get("role"),
            user_level=sender.get("level"),
            user_title=sender.get("title"),
            text=text,
            is_at_bot=is_at,
            message_id=data.get("message_id"),
        )

    @staticmethod
    def _extract_text(segments: list[dict[str, Any]]) -> str:
        parts = []
        for seg in segments:
            try:
                if seg.get("type") == "text":
                    parts.append(seg.get("data", {}).get("text", ""))
            except (AttributeError, TypeError):
                continue
        return "".join(parts)

    @property
    def has_images(self) -> bool:
        """Whether this message contains any image segments."""
        message_raw = self.raw.get("message")
        if not isinstance(message_raw, list):
            return False
        return any(seg.get("type") == "image" for seg in message_raw)

    @property
    def image_urls(self) -> list[str]:
        """All image URLs/file paths in this message."""
        message_raw = self.raw.get("message")
        if not isinstance(message_raw, list):
            return []
        urls: list[str] = []
        for seg in message_raw:
            if seg.get("type") == "image":
                data = seg.get("data", {})
                url = data.get("url") or data.get("file", "")
                if url:
                    urls.append(url)
        return urls


# ── Message builder ──────────────────────────────────────

class MessageBuilder:
    """Fluent builder for OneBot message segments."""

    def __init__(self) -> None:
        self._segments: list[dict[str, Any]] = []

    def text(self, content: str) -> MessageBuilder:
        if content:
            self._segments.append({"type": "text", "data": {"text": content}})
        return self

    def at(self, qq: str) -> MessageBuilder:
        self._segments.append({"type": "at", "data": {"qq": qq}})
        return self

    def image(self, file_path: str) -> MessageBuilder:
        self._segments.append({"type": "image", "data": {"file": file_path}})
        return self

    def reply(self, message_id: int) -> MessageBuilder:
        self._segments.append({"type": "reply", "data": {"id": str(message_id)}})
        return self

    def face(self, face_id: int) -> MessageBuilder:
        self._segments.append({"type": "face", "data": {"id": str(face_id)}})
        return self

    def record(self, file_path: str) -> MessageBuilder:
        """Send a voice/audio message (OneBot record type)."""
        self._segments.append({"type": "record", "data": {"file": file_path}})
        return self

    def music(self, music_type: str, song_id: str) -> MessageBuilder:
        """Send a music share card (OneBot music type). QQ native music share UI."""
        self._segments.append({
            "type": "music",
            "data": {
                "type": music_type,  # "163" for Netease, "qq" for QQ Music
                "id": str(song_id),
            },
        })
        return self

    def build(self) -> list[dict[str, Any]]:
        return self._segments


# ── LLBot Client ─────────────────────────────────────────

class LLBotClient:
    """Unified client for LLBot / OneBot HTTP API.

    Extensible: add new API methods by calling _post(endpoint, payload).
    """

    def __init__(
        self,
        api_url: str,
        token: str,
        timeout: int = 15,
        max_retries: int = 2,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.token = token
        self.timeout = timeout
        self._session = self._create_session(max_retries)

    def _create_session(self, max_retries: int) -> requests.Session:
        s = requests.Session()
        retry = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"POST", "GET"},
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"Authorization": f"Bearer {self.token}"})
        return s

    def _post(self, endpoint: str, payload: dict[str, Any]) -> bool:
        """Low-level POST. Returns True on success."""
        url = f"{self.api_url}/{endpoint}"
        try:
            r = self._session.post(url, json=payload, timeout=self.timeout)
            r.raise_for_status()
            logger.debug("LLBot %s OK", endpoint)
            return True
        except requests.exceptions.Timeout:
            logger.error("LLBot %s timeout", endpoint)
        except requests.exceptions.ConnectionError:
            logger.exception("LLBot %s connection error", endpoint)
        except requests.exceptions.HTTPError:
            logger.error("LLBot %s HTTP %s: %s", endpoint,
                         r.status_code if 'r' in dir() else '?',
                         (r.text[:200] if 'r' in dir() and r.text else ''))
        except Exception:
            logger.exception("LLBot %s unexpected error", endpoint)
        return False

    # ── Message sending ──────────────────────────────────

    def send_group_msg(
        self,
        group_id: str,
        message: list[dict[str, Any]] | MessageBuilder,
    ) -> bool:
        if isinstance(message, MessageBuilder):
            message = message.build()
        return self._post("send_group_msg", {
            "group_id": group_id,
            "message": message,
        })

    def send_private_msg(
        self,
        user_id: str,
        message: list[dict[str, Any]] | MessageBuilder,
    ) -> bool:
        if isinstance(message, MessageBuilder):
            message = message.build()
        return self._post("send_private_msg", {
            "user_id": user_id,
            "message": message,
        })

    # ── Convenience methods ──────────────────────────────

    def reply_to(
        self,
        msg: IncomingMessage,
        text: str,
    ) -> bool:
        """Reply with text. In groups: @user + text. In private: just text."""
        builder = MessageBuilder()
        if msg.msg_type == "group":
            if msg.message_id:
                builder.reply(msg.message_id)
            builder.at(msg.user_id).text(f" {text}")
            return self.send_group_msg(msg.group_id or "", builder)
        else:
            if msg.message_id:
                builder.reply(msg.message_id)
            builder.text(text)
            return self.send_private_msg(msg.user_id, builder)

    def reply_image(self, msg: IncomingMessage, image_path: str) -> bool:
        """Send an image as reply."""
        builder = MessageBuilder()
        if msg.msg_type == "group":
            if msg.message_id:
                builder.reply(msg.message_id)
            builder.image(image_path)
            return self.send_group_msg(msg.group_id or "", builder)
        else:
            builder.image(image_path)
            return self.send_private_msg(msg.user_id, builder)

    def send_text(self, msg: IncomingMessage, text: str) -> bool:
        """Send text without reply/at prefix. Useful for second messages."""
        builder = MessageBuilder().text(text)
        if msg.msg_type == "group":
            return self.send_group_msg(msg.group_id or "", builder)
        else:
            return self.send_private_msg(msg.user_id, builder)

    # ── Group operations ─────────────────────────────────

    def get_group_info(self, group_id: str) -> dict[str, Any] | None:
        try:
            r = self._session.post(
                f"{self.api_url}/get_group_info",
                json={"group_id": group_id},
                timeout=self.timeout,
            )
            return r.json().get("data", {})
        except Exception:
            logger.exception("get_group_info failed")
            return None

    def get_group_member_info(
        self, group_id: str, user_id: str
    ) -> dict[str, Any] | None:
        try:
            r = self._session.post(
                f"{self.api_url}/get_group_member_info",
                json={"group_id": group_id, "user_id": user_id},
                timeout=self.timeout,
            )
            return r.json().get("data", {})
        except Exception:
            logger.exception("get_group_member_info failed")
            return None

    # ── Group member list ───────────────────────────────

    def get_group_member_list(self, group_id: str) -> list[dict[str, Any]]:
        """Get all members of a group. Returns list of {user_id, nickname, role, ...}."""
        try:
            r = self._session.post(
                f"{self.api_url}/get_group_member_list",
                json={"group_id": group_id},
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json().get("data", [])
            return data if isinstance(data, list) else []
        except Exception:
            logger.exception("get_group_member_list failed for %s", group_id)
            return []

    # ── Extensibility ────────────────────────────────────

    def call(self, endpoint: str, payload: dict[str, Any]) -> bool:
        """Call any OneBot API endpoint. Use for future extensions."""
        return self._post(endpoint, payload)
