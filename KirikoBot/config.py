from __future__ import annotations

import os
from typing import Final

from dotenv import load_dotenv

load_dotenv()


class Config:
    ROBOT_QQ: Final[str | None] = os.getenv("ROBOT_QQ")
    ONEBOT_API: Final[str | None] = os.getenv("ONEBOT_API")
    ONEBOT_TOKEN: Final[str | None] = os.getenv("ONEBOT_TOKEN")
    DEEPSEEK_API: Final[str] = os.getenv("DEEPSEEK_API") or "https://api.deepseek.com/chat/completions"
    DEEPSEEK_TOKEN: Final[str | None] = os.getenv("DEEPSEEK_TOKEN")
    GROUP_ROLE: Final[str | None] = os.getenv("GROUP_ROLE")
    PRIVATE_ROLE: Final[str | None] = os.getenv("PRIVATE_ROLE")
    TAROT_ROLE: Final[str | None] = os.getenv("TAROT_ROLE")

    REQUEST_TIMEOUT: Final[int] = 30
    MAX_RETRIES: Final[int] = 3

    # ── Vision API (optional, for image description) ──────
    # Supports any OpenAI-compatible vision API endpoint.
    # Examples: SiliconFlow (Qwen-VL), Together AI, local vLLM, etc.
    # If not configured, image understanding falls back to context-based responses.
    VISION_API_URL: Final[str | None] = os.getenv("VISION_API_URL")
    VISION_API_KEY: Final[str | None] = os.getenv("VISION_API_KEY")
    VISION_MODEL: Final[str] = os.getenv("VISION_MODEL") or "Qwen/Qwen2-VL-7B-Instruct"

    @classmethod
    def validate(cls) -> None:
        required: dict[str, str | None] = {
            "ROBOT_QQ": cls.ROBOT_QQ,
            "ONEBOT_API": cls.ONEBOT_API,
            "ONEBOT_TOKEN": cls.ONEBOT_TOKEN,
            "DEEPSEEK_TOKEN": cls.DEEPSEEK_TOKEN,
            "GROUP_ROLE": cls.GROUP_ROLE,
            "PRIVATE_ROLE": cls.PRIVATE_ROLE,
            "TAROT_ROLE": cls.TAROT_ROLE,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                "Please check your .env file."
            )


Config.validate()
