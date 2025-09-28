from __future__ import annotations
import numpy as np, pandas as pd

def add_rolls(df: pd.DataFrame, col: str, wins=(3,7,14)) -> None:
    for w in wins:
        df[f"{col}_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
        df[f"{col}_sum_{w}"]  = df[col].rolling(w, min_periods=1).sum()

def build_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    df = daily_df.sort_values("time").copy()
    for col in ("precipitation_sum","rain_sum","temperature_2m_mean"):
        if col in df:
            add_rolls(df, col)
            df[f"{col}_lag1"] = df[col].shift(1)
            df[f"{col}_lag7"] = df[col].shift(7)
    doy = pd.to_datetime(df["time"]).dayofyear
    df["doy_sin"] = np.sin(2*np.pi*doy/366)
    df["doy_cos"] = np.cos(2*np.pi*doy/366)
    return df
