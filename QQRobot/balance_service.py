from __future__ import annotations

import logging
from typing import Any

import requests

from config import Config

logger = logging.getLogger(__name__)


class BalanceService:
    """Queries DeepSeek account balance via API."""

    TIMEOUT = 10

    @staticmethod
    def _create_session() -> requests.Session:
        session = requests.Session()
        from requests.adapters import HTTPAdapter, Retry
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods={"GET"},
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        return session

    def get_balance(self) -> dict[str, Any]:
        """Query DeepSeek balance. Returns dict with balance info or error."""
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {Config.DEEPSEEK_TOKEN}",
        }

        session = self._create_session()
        try:
            r = session.get(
                "https://api.deepseek.com/user/balance",
                headers=headers,
                timeout=self.TIMEOUT,
            )
            r.raise_for_status()
            data = r.json()
            logger.info("Balance queried successfully")
            return {"ok": True, "data": data}
        except requests.exceptions.Timeout:
            logger.error("Balance query timed out")
            return {"ok": False, "error": "查询超时，请稍后再试"}
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else None
            logger.error("Balance query HTTP %s", status)
            msg = {401: "API密钥无效", 403: "无权限访问"}.get(status or 0, f"HTTP {status}")
            return {"ok": False, "error": msg}
        except Exception:
            logger.exception("Balance query failed")
            return {"ok": False, "error": "查询失败，网络或服务异常"}

    def format_balance(self) -> str:
        """Query and format balance as a human-readable string."""
        result = self.get_balance()
        if not result["ok"]:
            return f"查询DeepSeek余额失败：{result['error']}"

        data = result["data"]
        # DeepSeek returns balance_infos array with currency, total_balance, etc.
        infos = data.get("balance_infos", [])
        if not infos:
            return "DeepSeek账户余额：暂无数据"

        info = infos[0]
        currency = info.get("currency", "CNY")
        total = info.get("total_balance", "未知")
        granted = info.get("granted_balance", "未知")
        topped_up = info.get("topped_up_balance", "未知")

        return (
            f"DeepSeek账户余额：\n"
            f"  总余额：{total} {currency}\n"
            f"  充值余额：{topped_up} {currency}\n"
            f"  赠送余额：{granted} {currency}"
        )
