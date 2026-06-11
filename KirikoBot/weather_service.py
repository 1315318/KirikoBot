from __future__ import annotations

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)


class WeatherService:
    """Fetches weather data from wttr.in (free, no API key needed)."""

    BASE_URL = "https://wttr.in"
    TIMEOUT = 10

    def get_weather(self, city: str) -> dict[str, Any] | None:
        """Fetch current weather and forecast for a city."""
        try:
            url = f"{self.BASE_URL}/{city}?format=j1&lang=zh"
            response = requests.get(url, timeout=self.TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            logger.error("Weather request timed out for: %s", city)
            return None
        except requests.exceptions.ConnectionError:
            logger.exception("Weather connection error for: %s", city)
            return None
        except requests.exceptions.HTTPError:
            logger.exception("Weather HTTP error for: %s", city)
            return None
        except Exception:
            logger.exception("Unexpected weather error for: %s", city)
            return None

        try:
            current = data["current_condition"][0]
            forecast = data["weather"][:3]  # next 3 days

            return {
                "city": city,
                "temp": current.get("temp_C", "?"),
                "feels_like": current.get("FeelsLikeC", "?"),
                "humidity": current.get("humidity", "?"),
                "weather_desc": current.get("weatherDesc", [{}])[0].get("value", "未知"),
                "wind_speed": current.get("windspeedKmph", "?"),
                "wind_dir": current.get("winddir16Point", "?"),
                "visibility": current.get("visibility", "?"),
                "forecast": [
                    {
                        "date": d.get("date", "?"),
                        "high": d.get("maxtempC", "?"),
                        "low": d.get("mintempC", "?"),
                        "desc": d.get("weatherDesc", [{}])[0].get("value", "未知") if d.get("weatherDesc") else "未知",
                    }
                    for d in forecast
                ],
            }
        except (KeyError, IndexError, TypeError):
            logger.exception("Failed to parse weather data for: %s", city)
            return None
