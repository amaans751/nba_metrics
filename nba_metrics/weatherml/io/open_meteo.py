from __future__ import annotations
from typing import Iterable, Dict, Any
import httpx
import pandas as pd

_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST = "https://api.open-meteo.com/v1/forecast"

def fetch_daily_archive(
    lat: float, lon: float, start_date: str, end_date: str,
    daily_vars: Iterable[str], timezone: str = "Australia/Sydney",
    timeout: float = 60.0
) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "daily": ",".join(daily_vars),
        "timezone": timezone,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.get(_ARCHIVE, params=params)
        r.raise_for_status()
    js: Dict[str, Any] = r.json()
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"]).dt.date
    return df

def fetch_forecast_past_days(
    lat: float, lon: float, past_days: int,
    daily_vars: Iterable[str], timezone: str = "Australia/Sydney",
    timeout: float = 60.0
) -> pd.DataFrame:
    params = {
        "latitude": lat, "longitude": lon,
        "past_days": int(past_days),
        "daily": ",".join(daily_vars),
        "timezone": timezone,
    }
    with httpx.Client(timeout=timeout) as client:
        r = client.get(_FORECAST, params=params)
        r.raise_for_status()
    js = r.json()
    df = pd.DataFrame(js["daily"])
    df["time"] = pd.to_datetime(df["time"]).dt.date
    return df
