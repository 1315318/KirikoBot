from __future__ import annotations

import json
import logging
import re
from typing import Any

import markdown
import requests
from requests.adapters import HTTPAdapter, Retry

from config import Config

logger = logging.getLogger(__name__)


class AiServer:
    def __init__(
        self,
        system_text: str,
        user_text: str,
        history_list: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        model_type: str = "deepseek-v4-flash",
        thinking_type: str = "disabled",
    ) -> None:
        if history_list is None:
            history_list = []
        if tools is None:
            tools = []

        self.system_text = system_text
        self.user_text = user_text
        self.model_type = model_type
        self.thinking_type = thinking_type
        self.history_list = history_list
        self.tools = tools

        self.ai_message: dict[str, Any] = {}
        self.ai_text: str = ""
        self.airesponse_tool_id: str = ""
        self.airesponse_tool_calls: list[dict[str, Any]] = []
        self.tool_results: list[dict[str, Any]] = []
        self.reasoning_content: str = ""

    @staticmethod
    def _create_session() -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=Config.MAX_RETRIES,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"POST"},
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def ai_request(self) -> None:
        # Reset to prevent stale tool_calls from leaking into history on error
        self.ai_message = {}
        self.ai_text = ""
        self.airesponse_tool_calls = []
        self.airesponse_tool_id = ""

        request_dict: dict[str, Any] = {
            "messages": [
                {"content": self.system_text, "role": "system"},
                *self.history_list,
                {"content": self.user_text, "role": "user"},
            ],
            "model": self.model_type,
            "thinking": {"type": self.thinking_type},
            "tools": self.tools,
            "tool_choice": "auto",
            "response_format": {"type": "text"},
            "max_tokens": 6000,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 0.9,
            "stop": None,
            "stream": False,
            "stream_options": None,
            "logprobs": False,
            "top_logprobs": None,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {Config.DEEPSEEK_TOKEN}",
        }

        session = self._create_session()

        try:
            logger.debug("DeepSeek request: %s", json.dumps(request_dict, ensure_ascii=False))
            response = session.post(
                Config.DEEPSEEK_API,
                headers=headers,
                data=json.dumps(request_dict),
                timeout=Config.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout:
            logger.error("DeepSeek API request timed out after %ss", Config.REQUEST_TIMEOUT)
            self.ai_text = "抱歉，AI思考时间有点长，请稍后再试~"
            return
        except requests.exceptions.ConnectionError:
            logger.exception("DeepSeek API connection error")
            self.ai_text = "抱歉，网络连接出现了问题，请稍后再试~"
            return
        except requests.exceptions.HTTPError as e:
            logger.error("DeepSeek API HTTP error: %s", e)
            status_code = e.response.status_code if e.response is not None else None
            # Log response body for debugging
            if e.response is not None:
                try:
                    logger.error("DeepSeek error body: %s", e.response.text[:500])
                except Exception:
                    pass
            if status_code == 401:
                self.ai_text = "抱歉，AI服务配置有问题，请联系管理员~"
            elif status_code == 429:
                self.ai_text = "抱歉，AI服务请求太频繁了，请稍后再试~"
            elif status_code in (500, 502, 503, 504):
                self.ai_text = "抱歉，AI服务暂时不可用，请稍后再试~"
            else:
                self.ai_text = f"抱歉，AI服务返回了异常状态({status_code})，请稍后再试~"
            return
        except Exception:
            logger.exception("Unexpected error during DeepSeek API request")
            self.ai_text = "抱歉，处理请求时遇到了问题，请稍后再试~"
            return

        try:
            response_data = response.json()
        except (json.JSONDecodeError, ValueError):
            logger.error("DeepSeek API returned non-JSON response: %s", response.text[:500])
            self.ai_text = "抱歉，AI服务返回了异常数据，请稍后再试~"
            return

        try:
            self.ai_message = response_data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            logger.error("Unexpected DeepSeek response structure: %s", json.dumps(response_data, ensure_ascii=False)[:500])
            if "error" in response_data:
                logger.error("DeepSeek API error: %s", response_data["error"])
            self.ai_text = "抱歉，AI服务返回了意外的数据格式，请稍后再试~"
            return

        # Capture thinking chain for logging/frontend display
        self.reasoning_content = self.ai_message.get("reasoning_content", "") or ""

        raw_content = self.ai_message.get("content", "")
        try:
            self.ai_text = self._format_for_qq(raw_content)
        except Exception:
            logger.exception("Failed to process markdown content")
            self.ai_text = raw_content

    def add_tool_result(self, tool_call_id: str, result: str) -> None:
        self.tool_results.append({"tool_call_id": tool_call_id, "content": result})

    def follow_up_request(self, original_tool_calls: list[dict[str, Any]]) -> None:
        """Send a follow-up request with tool call results injected into messages.
        This enables multi-turn conversation where the AI incorporates tool results."""
        # Capture original message fields before resetting. reasoning_content MUST be
        # passed back to the API when thinking mode is enabled, or DeepSeek returns 400.
        original_content = self.ai_message.get("content", "")
        original_reasoning = self.ai_message.get("reasoning_content", "")
        self.ai_message = {}
        self.ai_text = ""

        # Build messages: system + history + user + assistant(tool_calls) + tool_results
        assistant_msg: dict[str, Any] = {"role": "assistant", "tool_calls": original_tool_calls}
        if original_reasoning:
            assistant_msg["reasoning_content"] = original_reasoning
        if original_content:
            assistant_msg["content"] = original_content
        follow_up_messages: list[dict[str, Any]] = [
            {"content": self.system_text, "role": "system"},
            *self.history_list,
            {"content": self.user_text, "role": "user"},
            assistant_msg,
        ]
        for tr in self.tool_results:
            follow_up_messages.append({"role": "tool", **tr})

        request_dict: dict[str, Any] = {
            "messages": follow_up_messages,
            "model": self.model_type,
            "thinking": {"type": self.thinking_type},
            "max_tokens": 6000,
            "frequency_penalty": 0,
            "presence_penalty": 0,
            "temperature": 0,
            "top_p": 0.9,
            "stream": False,
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {Config.DEEPSEEK_TOKEN}",
        }

        session = self._create_session()

        try:
            logger.debug("Follow-up request: %s", json.dumps(request_dict, ensure_ascii=False))
            response = session.post(
                Config.DEEPSEEK_API,
                headers=headers,
                data=json.dumps(request_dict),
                timeout=Config.REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            response_data = response.json()
            self.ai_message = response_data["choices"][0]["message"]
            self.reasoning_content = self.ai_message.get("reasoning_content", "") or ""
            raw_content = self.ai_message.get("content", "")
            self.ai_text = self._format_for_qq(raw_content)
        except requests.exceptions.HTTPError as e:
            error_body = ""
            if e.response is not None:
                try:
                    error_body = e.response.text[:1000]
                except Exception:
                    pass
            logger.error("Follow-up HTTP %s: %s", e.response.status_code if e.response is not None else "?", error_body)
            self.ai_text = ""
        except Exception:
            logger.exception("Follow-up AI request failed")
            self.ai_text = ""

    @staticmethod
    def _format_for_qq(text: str) -> str:
        """Convert markdown to QQ-friendly plain text."""
        # Convert markdown to HTML then strip tags
        html = markdown.markdown(text, extensions=["extra"])
        plain = re.sub(r"<[^>]+>", "", html)

        # Replace markdown tables with simple list format
        plain = re.sub(r"^\|.*\|$", lambda m: m.group().replace("|", " · "), plain, flags=re.MULTILINE)
        # Collapse excessive newlines
        plain = re.sub(r"\n{3,}", "\n\n", plain)
        # Remove horizontal rules
        plain = re.sub(r"^[-*_]{3,}\s*$", "", plain, flags=re.MULTILINE)

        # Truncate if still too long for QQ
        if len(plain) > 2500:
            plain = plain[:2400] + "\n\n...（内容过长已截断）"

        return plain.strip()
